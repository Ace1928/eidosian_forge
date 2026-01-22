from __future__ import annotations
import gzip
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.io import zopen
from monty.json import MontyDecoder, MSONable
from pymatgen.core import Lattice, Molecule, Structure
@dataclass
class AimsGeometryIn(MSONable):
    """Representation of an aims geometry.in file

    Attributes:
        _content (str): The content of the input file
        _structure (Structure | Molecule): The structure or molecule
            representation of the file
    """
    _content: str
    _structure: Structure | Molecule

    @classmethod
    def from_str(cls, contents: str) -> Self:
        """Create an input from the content of an input file

        Args:
            contents (str): The content of the file

        Returns:
            The AimsGeometryIn file for the string contents
        """
        content_lines = [line.strip() for line in contents.split('\n') if len(line.strip()) > 0 and line.strip()[0] != '#']
        species, coords, is_frac, lattice_vectors = ([], [], [], [])
        charges_dct, moments_dct = ({}, {})
        for line in content_lines:
            inp = line.split()
            if inp[0] == 'atom' or inp[0] == 'atom_frac':
                coords.append([float(ii) for ii in line.split()[1:4]])
                species.append(inp[4])
                is_frac.append(inp[0] == 'atom_frac')
            if inp[0] == 'lattice_vector':
                lattice_vectors.append([float(ii) for ii in line.split()[1:4]])
            if inp[0] == 'initial_moment':
                moments_dct[len(coords) - 1] = float(inp[1])
            if inp[0] == 'initial_charge':
                charges_dct[len(coords) - 1] = float(inp[1])
        charge = np.zeros(len(coords))
        for key, val in charges_dct.items():
            charge[key] = val
        magmom = np.zeros(len(coords))
        for key, val in moments_dct.items():
            magmom[key] = val
        if len(lattice_vectors) == 3:
            lattice = Lattice(lattice_vectors)
            for cc in range(len(coords)):
                if is_frac[cc]:
                    coords[cc] = lattice.get_cartesian_coords(np.array(coords[cc]).reshape(1, 3)).flatten()
        elif len(lattice_vectors) == 0:
            lattice = None
            if any(is_frac):
                raise ValueError('Fractional coordinates given in file with no lattice vectors.')
        else:
            raise ValueError('Incorrect number of lattice vectors passed.')
        site_props = {'magmom': magmom, 'charge': charge}
        if lattice is None:
            structure = Molecule(species, coords, np.sum(charge), site_properties=site_props)
        else:
            structure = Structure(lattice, species, coords, np.sum(charge), coords_are_cartesian=True, site_properties=site_props)
        return cls(_content='\n'.join(content_lines), _structure=structure)

    @classmethod
    def from_file(cls, filepath: str | Path) -> Self:
        """Create an AimsGeometryIn from an input file.

        Args:
            filepath (str | Path): The path to the input file (either plain text of gzipped)

        Returns:
            AimsGeometryIn: The input object represented in the file
        """
        with zopen(filepath, mode='rt') as in_file:
            content = in_file.read()
        return cls.from_str(content)

    @classmethod
    def from_structure(cls, structure: Structure | Molecule) -> Self:
        """Construct an input file from an input structure.

        Args:
            structure (Structure | Molecule): The structure for the file

        Returns:
            AimsGeometryIn: The input object for the structure
        """
        content_lines: list[str] = []
        if isinstance(structure, Structure):
            for lv in structure.lattice.matrix:
                content_lines.append(f'lattice_vector {lv[0]: .12e} {lv[1]: .12e} {lv[2]: .12e}')
        charges = structure.site_properties.get('charge', np.zeros(len(structure.species)))
        magmoms = structure.site_properties.get('magmom', np.zeros(len(structure.species)))
        for species, coord, charge, magmom in zip(structure.species, structure.cart_coords, charges, magmoms):
            content_lines.append(f'atom {coord[0]: .12e} {coord[1]: .12e} {coord[2]: .12e} {species}')
            if charge != 0:
                content_lines.append(f'     initial_charge {charge:.12e}')
            if magmom != 0:
                content_lines.append(f'     initial_moment {magmom:.12e}')
        return cls(_content='\n'.join(content_lines), _structure=structure)

    @property
    def structure(self) -> Structure | Molecule:
        """Access structure for the file"""
        return self._structure

    @property
    def content(self) -> str:
        """Access the contents of the file"""
        return self._content

    def write_file(self, directory: str | Path | None=None, overwrite: bool=False) -> None:
        """Write the geometry.in file

        Args:
            directory (str | Path | None): The directory to write the geometry.in file
            overwrite (bool): If True allow to overwrite existing files
        """
        directory = directory or Path.cwd()
        if not overwrite and (Path(directory) / 'geometry.in').exists():
            raise ValueError(f'geometry.in file exists in {directory}')
        with open(f'{directory}/geometry.in', mode='w') as file:
            file.write(f'#{'=' * 72}\n')
            file.write(f'# FHI-aims geometry file: {directory}/geometry.in\n')
            file.write('# File generated from pymatgen\n')
            file.write(f'# {time.asctime()}\n')
            file.write(f'#{'=' * 72}\n')
            file.write(self.content)
            file.write('\n')

    def as_dict(self) -> dict[str, Any]:
        """Get a dictionary representation of the geometry.in file."""
        dct = {}
        dct['@module'] = type(self).__module__
        dct['@class'] = type(self).__name__
        dct['content'] = self.content
        dct['structure'] = self.structure
        return dct

    @classmethod
    def from_dict(cls, dct: dict[str, Any]) -> Self:
        """Initialize from dictionary.

        Args:
            dct (dict[str, Any]): The MontyEncoded dictionary of the AimsGeometryIn object

        Returns:
            The input object represented by the dict
        """
        decoded = {key: MontyDecoder().process_decoded(val) for key, val in dct.items() if not key.startswith('@')}
        return cls(_content=decoded['content'], _structure=decoded['structure'])