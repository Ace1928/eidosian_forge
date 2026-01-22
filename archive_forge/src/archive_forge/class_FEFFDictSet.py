from __future__ import annotations
import abc
import logging
import os
import sys
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from monty.os.path import zpath
from monty.serialization import loadfn
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.feff.inputs import Atoms, Header, Potential, Tags
class FEFFDictSet(AbstractFeffInputSet):
    """
    Standard implementation of FeffInputSet, which can be extended by specific
    implementations.
    """

    def __init__(self, absorbing_atom: str | int, structure: Structure | Molecule, radius: float, config_dict: dict, edge: str='K', spectrum: str='EXAFS', nkpts=1000, user_tag_settings: dict | None=None, spacegroup_analyzer_settings: dict | None=None):
        """
        Args:
            absorbing_atom (str/int): absorbing atom symbol or site index
            structure: Structure or Molecule object. If a Structure, SpaceGroupAnalyzer is used to
                determine symmetrically-equivalent sites. If a Molecule, there is no symmetry
                checking.
            radius (float): cluster radius
            config_dict (dict): control tag settings dict
            edge (str): absorption edge
            spectrum (str): type of spectrum to calculate, available options :
                EXAFS, XANES, DANES, XMCD, ELNES, EXELFS, FPRIME, NRIXS, XES.
                The default is EXAFS.
            nkpts (int): Total number of kpoints in the brillouin zone. Used
                only when feff is run in the reciprocal space mode.
            user_tag_settings (dict): override default tag settings. To delete
                tags, set the key '_del' in the user_tag_settings.
                eg: user_tag_settings={"_del": ["COREHOLE", "EXCHANGE"]}
                To specify a net charge on the structure, pass an "IONS" tag containing a list
                    of tuples where the first element is the unique potential value (ipot value)
                    and the second element is the charge to be applied to atoms associated
                    with that potential, e.g. {"IONS": [(0, 0.1), (1, 0.1), (2, 0.1)]}
                    will result in.

                    ION 0 0.1
                    ION 1 0.1
                    ION 2 0.1

                    being written to the input file.
            spacegroup_analyzer_settings (dict): parameters passed to SpacegroupAnalyzer.
                E.g., {"symprec": 0.01, "angle_tolerance": 4}
        """
        self.absorbing_atom = absorbing_atom
        self.user_tag_settings = user_tag_settings or {}
        if structure.is_ordered:
            if isinstance(structure, Structure):
                if structure.charge != 0:
                    raise ValueError('Structure objects with a net charge are not supported!')
            elif isinstance(structure, Molecule):
                if structure.charge != 0 and (not self.user_tag_settings.get('IONS')):
                    warnings.warn('For Molecule objects with a net charge it is recommended to set one or more ION tags in the input file by modifying user_tag_settings. Consult the FEFFDictSet docstring and the FEFF10 User Guide for more information.', UserWarning)
            else:
                raise ValueError("'structure' argument must be a Structure or Molecule!")
        else:
            raise ValueError('Structure with partial occupancies cannot be converted into atomic coordinates!')
        self.structure = structure
        self.radius = radius
        self.config_dict = deepcopy(config_dict)
        self.edge = edge
        self.spectrum = spectrum
        self.nkpts = nkpts
        self.config_dict['EDGE'] = self.edge
        self.config_dict.update(self.user_tag_settings)
        if '_del' in self.user_tag_settings:
            for tag in self.user_tag_settings['_del']:
                if tag in self.config_dict:
                    del self.config_dict[tag]
            del self.config_dict['_del']
        self.small_system = len(self.structure) < 14 and 'EXAFS' not in self.config_dict
        self.spacegroup_analyzer_settings = spacegroup_analyzer_settings or {}

    def header(self, source: str='', comment: str=''):
        """
        Creates header string from structure object.

        Args:
            source: Source identifier used to create structure, can be defined
                however user wants to organize structures, calculations, etc.
                example would be Materials Project material ID number.
            comment: comment to include in header

        Returns:
            Header
        """
        return Header(self.structure, source, comment, spacegroup_analyzer_settings=self.spacegroup_analyzer_settings)

    @property
    def tags(self) -> Tags:
        """
        FEFF job parameters.

        Returns:
            Tags
        """
        if 'RECIPROCAL' in self.config_dict:
            if self.small_system:
                self.config_dict['CIF'] = f'{self.structure.formula.replace(' ', '')}.cif'
                self.config_dict['TARGET'] = self.atoms.center_index + 1
                self.config_dict['COREHOLE'] = 'RPA'
                logger.warning('Setting COREHOLE = RPA for K-space calculation')
                if not self.config_dict.get('KMESH'):
                    abc = self.structure.lattice.abc
                    mult = (self.nkpts * abc[0] * abc[1] * abc[2]) ** (1 / 3)
                    self.config_dict['KMESH'] = [int(round(mult / length)) for length in abc]
            else:
                logger.warning('Large system(>=14 atoms) or EXAFS calculation, removing K-space settings')
                del self.config_dict['RECIPROCAL']
                self.config_dict.pop('CIF', None)
                self.config_dict.pop('TARGET', None)
                self.config_dict.pop('KMESH', None)
                self.config_dict.pop('STRFAC', None)
        return Tags(self.config_dict)

    @property
    def potential(self) -> Potential:
        """
        FEFF potential.

        Returns:
            Potential
        """
        return Potential(self.structure, self.absorbing_atom)

    @property
    def atoms(self) -> Atoms:
        """
        absorber + the rest.

        Returns:
            Atoms
        """
        return Atoms(self.structure, self.absorbing_atom, self.radius)

    def __str__(self):
        output = [self.spectrum]
        output.extend([f'{k} = {v}' for k, v in self.config_dict.items()])
        output.append('')
        return '\n'.join(output)

    @classmethod
    def from_directory(cls, input_dir: str) -> Self:
        """
        Read in a set of FEFF input files from a directory, which is
        useful when existing FEFF input needs some adjustment.
        """
        sub_d: dict = {'header': Header.from_file(zpath(os.path.join(input_dir, 'HEADER'))), 'parameters': Tags.from_file(zpath(os.path.join(input_dir, 'PARAMETERS')))}
        absorber_index = []
        radius = None
        feff_inp = zpath(f'{input_dir}/feff.inp')
        if 'RECIPROCAL' not in sub_d['parameters']:
            input_atoms = Atoms.cluster_from_file(feff_inp)
            shell_species = np.array([x.species_string for x in input_atoms])
            distance_matrix = input_atoms.distance_matrix[0, :]
            from math import ceil
            radius = int(ceil(input_atoms.get_distance(input_atoms.index(input_atoms[0]), input_atoms.index(input_atoms[-1]))))
            for site_index, site in enumerate(sub_d['header'].struct):
                if site.specie == input_atoms[0].specie:
                    site_atoms = Atoms(sub_d['header'].struct, absorbing_atom=site_index, radius=radius)
                    site_distance = np.array(site_atoms.get_lines())[:, 5].astype(float)
                    site_shell_species = np.array(site_atoms.get_lines())[:, 4]
                    shell_overlap = min(shell_species.shape[0], site_shell_species.shape[0])
                    if np.allclose(distance_matrix[:shell_overlap], site_distance[:shell_overlap]) and np.all(site_shell_species[:shell_overlap] == shell_species[:shell_overlap]):
                        absorber_index.append(site_index)
        if 'RECIPROCAL' in sub_d['parameters']:
            absorber_index = sub_d['parameters']['TARGET']
            absorber_index[0] = int(absorber_index[0]) - 1
        if 'XANES' in sub_d['parameters']:
            CONFIG = loadfn(f'{MODULE_DIR}/MPXANESSet.yaml')
            if radius is None:
                radius = 10
            return cls(absorber_index[0], sub_d['header'].struct, radius=radius, config_dict=CONFIG, edge=sub_d['parameters']['EDGE'], nkpts=1000, user_tag_settings=sub_d['parameters'])
        raise ValueError('Bad input directory.')