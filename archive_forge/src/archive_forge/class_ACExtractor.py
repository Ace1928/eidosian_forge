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
class ACExtractor(ACExtractorBase):
    """Extract information contained in atom.config : number of atoms, lattice, types, frac_coords, magmoms"""

    def __init__(self, file_path: PathLike) -> None:
        """Initialization function

        Args
            file_path (str): The absolute path of atom.config file.
        """
        self.atom_config_path = file_path
        self.n_atoms = self.get_n_atoms()
        self.lattice = self.get_lattice()
        self.types = self.get_types()
        self.coords = self.get_coords()
        self.magmoms = self.get_magmoms()

    def get_n_atoms(self) -> int:
        """Return the number of atoms in the structure."""
        first_row = linecache.getline(str(self.atom_config_path), 1)
        return int(first_row.split()[0])

    def get_lattice(self) -> np.ndarray:
        """Return the lattice of structure.

        Returns:
            lattice: np.ndarray, shape = (9,)
        """
        basis_vectors: list[float] = []
        content: str = 'LATTICE'
        idx_row: int = LineLocator.locate_all_lines(file_path=self.atom_config_path, content=content)[0]
        for row_idx in [idx_row + 1, idx_row + 2, idx_row + 3]:
            row_content: list[str] = linecache.getline(str(self.atom_config_path), row_idx).split()[:3]
            for value in row_content:
                basis_vectors.append(float(value))
        return np.array(basis_vectors)

    def get_types(self) -> np.ndarray:
        """Return the atomic number of atoms in structure.

        Returns:
            np.ndarray: Atomic numbers in order corresponding to sites
        """
        content = 'POSITION'
        idx_row = LineLocator.locate_all_lines(file_path=self.atom_config_path, content=content)[0]
        with open(self.atom_config_path) as file:
            atom_config_content = file.readlines()
        atomic_numbers_content = atom_config_content[idx_row:idx_row + self.n_atoms]
        atomic_numbers_lst = [int(row.split()[0]) for row in atomic_numbers_content]
        return np.array(atomic_numbers_lst)

    def get_coords(self) -> np.ndarray:
        """Return the fractional coordinates in structure.

        Returns:
            np.ndarray: Fractional coordinates.
        """
        coords_lst: list[np.ndarray] = []
        content: str = 'POSITION'
        idx_row: int = LineLocator.locate_all_lines(file_path=self.atom_config_path, content=content)[0]
        with open(self.atom_config_path) as file:
            atom_config_content = file.readlines()
        "\n        row_content:\n            '29         0.377262291145329         0.128590184800933         0.257759805813488     1  1  1'\n        "
        for row_content in atom_config_content[idx_row:idx_row + self.n_atoms]:
            row_content_lst = row_content.split()
            coord_tmp = [float(value) for value in row_content_lst[1:4]]
            coords_lst.append(np.array(coord_tmp))
        return np.array(coords_lst).reshape(-1)

    def get_magmoms(self) -> np.ndarray:
        """Return the magenetic moments of atoms in structure.

        Returns:
            np.ndarray: The magnetic moments of individual atoms.
        """
        content: str = 'MAGNETIC'
        magnetic_moments_lst: list[float] = []
        try:
            idx_row = LineLocator.locate_all_lines(file_path=self.atom_config_path, content=content)[-1]
            with open(self.atom_config_path) as file:
                atom_config_content = file.readlines()
            magnetic_moments_content = atom_config_content[idx_row:idx_row + self.n_atoms]
            magnetic_moments_lst = [float(tmp_magnetic_moment.split()[-1]) for tmp_magnetic_moment in magnetic_moments_content]
        except Exception:
            magnetic_moments_lst = [0 for _ in range(self.n_atoms)]
        return np.array(magnetic_moments_lst)