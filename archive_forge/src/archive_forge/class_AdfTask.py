from __future__ import annotations
import os
import re
from typing import TYPE_CHECKING
from monty.io import reverse_readline
from monty.itertools import chunks
from monty.json import MSONable
from monty.serialization import zopen
from pymatgen.core.structure import Molecule
class AdfTask(MSONable):
    """
    Basic task for ADF. All settings in this class are independent of molecules.

    Notes:
        Unlike other quantum chemistry packages (NWChem, Gaussian, ...),
        ADF does not support calculating force/gradient.
    """
    operations = dict(energy='Evaluate the single point energy.', optimize='Minimize the energy by varying the molecular structure.', frequencies='Compute second derivatives and print out an analysis of molecular vibrations.', freq='Same as frequencies.', numerical_frequencies='Compute molecular frequencies using numerical method.')

    def __init__(self, operation='energy', basis_set=None, xc=None, title='ADF_RUN', units=None, geo_subkeys=None, scf=None, other_directives=None):
        """
        Initialization method.

        Args:
            operation (str): The target operation.
            basis_set (AdfKey): The basis set definitions for this task. Defaults to 'DZ/Large'.
            xc (AdfKey): The exchange-correlation functionals. Defaults to PBE.
            title (str): The title of this ADF task.
            units (AdfKey): The units. Defaults to Angstroms/Degree.
            geo_subkeys (Sized): The subkeys for the block key 'GEOMETRY'.
            scf (AdfKey): The scf options.
            other_directives (Sized): User-defined directives.
        """
        if operation not in self.operations:
            raise AdfInputError(f'Invalid ADF task {operation}')
        self.operation = operation
        self.title = title
        self.basis_set = basis_set if basis_set is not None else self.get_default_basis_set()
        self.xc = xc if xc is not None else self.get_default_xc()
        self.units = units if units is not None else self.get_default_units()
        self.scf = scf if scf is not None else self.get_default_scf()
        self.other_directives = other_directives if other_directives is not None else []
        self._setup_task(geo_subkeys)

    @staticmethod
    def get_default_basis_set():
        """Returns: Default basis set."""
        return AdfKey.from_str('Basis\ntype DZ\ncore small\nEND')

    @staticmethod
    def get_default_scf():
        """Returns: ADF using default SCF."""
        return AdfKey.from_str('SCF\niterations 300\nEND')

    @staticmethod
    def get_default_geo():
        """Returns: ADFKey using default geometry."""
        return AdfKey.from_str('GEOMETRY SinglePoint\nEND')

    @staticmethod
    def get_default_xc():
        """Returns: ADFKey using default XC."""
        return AdfKey.from_str('XC\nGGA PBE\nEND')

    @staticmethod
    def get_default_units():
        """Returns: Default units."""
        return AdfKey.from_str('Units\nlength angstrom\nangle degree\nEnd')

    def _setup_task(self, geo_subkeys):
        """
        Setup the block 'Geometry' given subkeys and the task.

        Args:
            geo_subkeys (Sized): User-defined subkeys for the block 'Geometry'.

        Notes:
            Most of the run types of ADF are specified in the Geometry
            block except the 'AnalyticFreq'.
        """
        self.geo = AdfKey('Geometry', subkeys=geo_subkeys)
        if self.operation.lower() == 'energy':
            self.geo.add_option('SinglePoint')
            if self.geo.has_subkey('Frequencies'):
                self.geo.remove_subkey('Frequencies')
        elif self.operation.lower() == 'optimize':
            self.geo.add_option('GeometryOptimization')
            if self.geo.has_subkey('Frequencies'):
                self.geo.remove_subkey('Frequencies')
        elif self.operation.lower() == 'numerical_frequencies':
            self.geo.add_subkey(AdfKey('Frequencies'))
        else:
            self.other_directives.append(AdfKey('AnalyticalFreq'))
            if self.geo.has_subkey('Frequencies'):
                self.geo.remove_subkey('Frequencies')

    def __str__(self):
        out = f'TITLE {self.title}\n\n{self.units}\n{self.xc}\n{self.basis_set}\n{self.scf}\n{self.geo}'
        out += '\n'
        for block_key in self.other_directives:
            if not isinstance(block_key, AdfKey):
                raise ValueError(f'{block_key} is not an AdfKey!')
            out += str(block_key) + '\n'
        return out

    def as_dict(self):
        """A JSON-serializable dict representation of self."""
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'operation': self.operation, 'title': self.title, 'xc': self.xc.as_dict(), 'basis_set': self.basis_set.as_dict(), 'units': self.units.as_dict(), 'scf': self.scf.as_dict(), 'geo': self.geo.as_dict(), 'others': [k.as_dict() for k in self.other_directives]}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Construct a MSONable AdfTask object from the JSON dict.

        Args:
            dct: A dict of saved attributes.

        Returns:
            An AdfTask object recovered from the JSON dict ``d``.
        """

        def _from_dict(_d):
            return AdfKey.from_dict(_d) if _d is not None else None
        operation = dct.get('operation')
        title = dct.get('title')
        basis_set = _from_dict(dct.get('basis_set'))
        xc = _from_dict(dct.get('xc'))
        units = _from_dict(dct.get('units'))
        scf = _from_dict(dct.get('scf'))
        others = [AdfKey.from_dict(o) for o in dct.get('others', [])]
        geo = _from_dict(dct.get('geo'))
        return cls(operation, basis_set, xc, title, units, geo.subkeys, scf, others)