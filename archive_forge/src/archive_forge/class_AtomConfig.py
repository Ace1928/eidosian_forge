from __future__ import annotations
import linecache
from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.kpath import KPathSeek
class AtomConfig(MSONable):
    """Object for representing the data in a atom.config or final.config file."""

    def __init__(self, structure: Structure, sort_structure: bool=False):
        """Initialization function.

        Args:
            structure (Structure): Structure object
            sort_structure (bool, optional): Whether to sort the structure. Useful if species
                are not grouped properly together. Defaults to False.
        """
        self.structure: Structure = structure
        if sort_structure:
            self.structure = self.structure.get_sorted_structure()
        elements_counter = dict(sorted(Counter(self.structure.species).items()))
        true_names = [f'{tmp_key}{tmp_value}' for tmp_key, tmp_value in elements_counter.items()]
        self.true_names = ''.join(true_names)

    def __repr__(self):
        return self.get_str()

    def __str__(self):
        return self.get_str()

    @classmethod
    def from_str(cls, data: str, mag: bool=False) -> Self:
        """Reads a atom.config from a string.

        Args:
            data (str): string containing atom.config data
            mag (bool, optional): Whether to read magnetic moment information.

        Returns:
            AtomConfig object
        """
        ac_extractor = ACstrExtractor(atom_config_str=data)
        properties: dict[str, float] = {}
        structure = Structure(lattice=ac_extractor.get_lattice(), species=ac_extractor.get_types(), coords=ac_extractor.get_coords().reshape(-1, 3), coords_are_cartesian=False, properties=properties)
        if mag:
            magmoms = ac_extractor.get_magmoms()
            for idx, tmp_site in enumerate(structure):
                tmp_site.properties.update({'magmom': magmoms[idx]})
        return cls(structure)

    @classmethod
    def from_file(cls, filename: PathLike, mag: bool=False) -> Self:
        """Returns a AtomConfig from a file

        Args:
            filename (PathLike): File name containing AtomConfig data
            mag (bool, optional): Whether to read magnetic moments. Defaults to True.

        Returns:
            AtomConfig object.
        """
        with zopen(filename, 'rt') as file:
            return cls.from_str(data=file.read(), mag=mag)

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Returns a AtomConfig object from a dictionary.

        Args:
            dct: dict containing atom.config data

        Returns:
            AtomConfig object.
        """
        return cls(Structure.from_dict(dct['structure']))

    def get_str(self) -> str:
        """Return a string describing the structure in atom.config format.

        Returns:
            str: String representation of atom.config
        """
        lattice = self.structure.lattice
        if np.linalg.det(lattice.matrix) < 0:
            lattice = Lattice(-lattice.matrix)
        lines: list[str] = []
        lines.append(f'\t{self.structure.num_sites} atoms\n')
        lines.append('Lattice vector\n')
        for ii in range(3):
            lines.append(f'{lattice.matrix[ii][0]:>15f}{lattice.matrix[ii][1]:>15f}{lattice.matrix[ii][2]:>15f}\n')
        lines.append('Position, move_x, move_y, move_z\n')
        for ii in range(self.structure.num_sites):
            lines.append(f'{int(self.structure.species[ii].Z):>4d}')
            lines.append(f'{self.structure.frac_coords[ii][0]:>15f}')
            lines.append(f'{self.structure.frac_coords[ii][1]:>15f}')
            lines.append(f'{self.structure.frac_coords[ii][2]:>15f}')
            lines.append('   1   1   1\n')
        if 'magmom' in self.structure.sites[0].properties:
            lines.append('MAGNETIC\n')
            for _, tmp_site in enumerate(self.structure.sites):
                lines.append(f'{int(tmp_site.specie.Z):>4d}{tmp_site.properties['magmom']:>15f}\n')
        return ''.join(lines)

    def write_file(self, filename: PathLike, **kwargs):
        """Writes AtomConfig to a file."""
        with zopen(filename, 'wt') as file:
            file.write(self.get_str(**kwargs))

    def as_dict(self):
        """
        Returns:
            dict
        """
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'structure': self.structure.as_dict(), 'true_names': self.true_names}