from __future__ import annotations
import re
from typing import TYPE_CHECKING, no_type_check
import numpy as np
from monty.io import zopen
from pymatgen.core.structure import Structure
from pymatgen.core.units import Ry_to_eV, bohr_to_angstrom
from pymatgen.electronic_structure.core import Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.num import round_to_sigfigs
class LMTOCopl:
    """Class for reading COPL files, which contain COHP data.

    Attributes:
        cohp_data (dict): Contains the COHP data of the form:
            {bond: {"COHP": {Spin.up: cohps, Spin.down:cohps},
                    "ICOHP": {Spin.up: icohps, Spin.down: icohps},
                    "length": bond length}
        efermi (float): The Fermi energy in Ry or eV.
        energies (list): Sequence of energies in Ry or eV.
        is_spin_polarized (bool): Boolean to indicate if the calculation is spin polarized.
    """

    def __init__(self, filename='COPL', to_eV=False):
        """
        Args:
            filename: filename of the COPL file. Defaults to "COPL".
            to_eV: LMTO-ASA gives energies in Ry. To convert energies into
              eV, set to True. Defaults to False for energies in Ry.
        """
        with zopen(filename, mode='rt') as file:
            contents = file.read().split('\n')[:-1]
        parameters = contents[1].split()
        num_bonds = int(parameters[0])
        if int(parameters[1]) == 2:
            spins = [Spin.up, Spin.down]
            self.is_spin_polarized = True
        else:
            spins = [Spin.up]
            self.is_spin_polarized = False
        data = np.array([np.array(row.split(), dtype=float) for row in contents[num_bonds + 2:]]).transpose()
        if to_eV:
            self.energies = np.array([round_to_sigfigs(energy, 5) for energy in data[0] * Ry_to_eV], dtype=float)
            self.efermi = round_to_sigfigs(float(parameters[-1]) * Ry_to_eV, 5)
        else:
            self.energies = data[0]
            self.efermi = float(parameters[-1])
        cohp_data = {}
        for bond in range(num_bonds):
            label, length, sites = self._get_bond_data(contents[2 + bond])
            cohp = {spin: data[2 * (bond + s * num_bonds) + 1] for s, spin in enumerate(spins)}
            if to_eV:
                icohp = {spin: np.array([round_to_sigfigs(i, 5) for i in data[2 * (bond + s * num_bonds) + 2] * Ry_to_eV]) for s, spin in enumerate(spins)}
            else:
                icohp = {spin: data[2 * (bond + s * num_bonds) + 2] for s, spin in enumerate(spins)}
            if label in cohp_data:
                idx = 1
                lab = f'{label}-{idx}'
                while lab in cohp_data:
                    idx += 1
                    lab = f'{label}-{idx}'
                label = lab
            cohp_data[label] = {'COHP': cohp, 'ICOHP': icohp, 'length': length, 'sites': sites}
        self.cohp_data = cohp_data

    @staticmethod
    def _get_bond_data(line):
        """
        Subroutine to extract bond label, site indices, and length from
        a COPL header line. The site indices are zero-based, so they
        can be easily used with a Structure object.

        Example header line: Fe-1/Fe-1-tr(-1,-1,-1) : 2.482 Ang.

        Args:
            line: line in the COHPCAR header describing the bond.

        Returns:
            The bond label, the bond length and a tuple of the site indices.
        """
        line = line.split()
        length = float(line[2])
        sites = line[0].replace('/', '-').split('-')
        site_indices = tuple((int(ind) - 1 for ind in sites[1:4:2]))
        species = tuple((re.split('\\d+', spec)[0] for spec in sites[0:3:2]))
        label = f'{species[0]}{site_indices[0] + 1}-{species[1]}{site_indices[1] + 1}'
        return (label, length, site_indices)