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
class HighSymmetryPoint(MSONable):
    """HighSymmetryPoint object for reading and writing HIGH_SYMMETRY_POINTS file which generate line-mode kpoints."""

    def __init__(self, reciprocal_lattice: np.ndarray, kpts: dict[str, list], path: list[list[str]], density: float):
        """Initialization function.

        Args:
            reciprocal_lattice (np.array): Reciprocal lattice.
            kpts (dict[str, list[float]]): Kpoints and their corresponding fractional coordinates.
            path (list[list[str]]): All k-paths, with each list representing one k-path.
            density (float): Density of kpoints mesh with factor of 2*pi.
        """
        self.reciprocal_lattice: np.ndarray = reciprocal_lattice
        self.kpath: dict = {}
        self.kpath.update({'kpoints': kpts})
        self.kpath.update({'path': path})
        self.density = density

    @classmethod
    def from_structure(cls, structure: Structure, dim: int, density: float=0.01) -> Self:
        """Obtain HighSymmetry object from Structure object.

        Args:
            structure (Structure): A structure object.
            dim (int): Dimension of the material system (2 or 3).
            density (float, optional): Density of kpoints mesh without factor of 2*pi. Defaults to 0.01.
                The program will automatically convert it to with factor of 2*pi.
        """
        reciprocal_lattice: np.ndarray = structure.lattice.reciprocal_lattice.matrix
        gen_kpt = GenKpt.from_structure(structure=structure, dim=dim, density=density)
        return cls(reciprocal_lattice, gen_kpt.kpath['kpoints'], gen_kpt.kpath['path'], density * 2 * np.pi)

    def get_str(self) -> str:
        """Returns a string describing high symmetry points in HIGH_SYMMETRY_POINTS format."""

        def calc_distance(hsp1: str, hsp2: str) -> float:
            """Calculate the distance of two high symmetry points.

            Returns:
                distance (float): Calculate the distance of two high symmetry points. With factor of 2*pi.
            """
            hsp1_coord: np.ndarray = np.dot(np.array(self.kpath['kpoints'][hsp1]).reshape(1, 3), self.reciprocal_lattice)
            hsp2_coord: np.ndarray = np.dot(np.array(self.kpath['kpoints'][hsp2]).reshape(1, 3), self.reciprocal_lattice)
            return float(np.linalg.norm(hsp2_coord - hsp1_coord))

        def get_hsp_row_str(label: str, index: int, coordinate: float) -> str:
            """
            Return string containing name, index, coordinate of the certain high symmetry point
            in HIGH_SYMMETRY_POINTS format.

            Args:
                label (str): Name of the high symmetry point.
                index (int): Index of the high symmetry point.
                coordinate (float): Coordinate in bandstructure of the high symmetry point.

            Returns:
                str: String containing name, index, coordinate of the certain high symmetry point
                    in HIGH_SYMMETRY_POINTS format.
            """
            if label == 'GAMMA':
                return f'G            {index:>4d}         {coordinate:>.6f}\n'
            return f'{label}            {index:>4d}         {coordinate:>.6f}\n'
        discontinue_pairs: list[list[str]] = []
        for ii in range(len(self.kpath['path']) - 1):
            discontinue_pairs.append([self.kpath['path'][ii][-1], self.kpath['path'][ii + 1][0]])
        flatten_paths: list[str] = [tmp_hsp for tmp_path in self.kpath['path'] for tmp_hsp in tmp_path]
        index: int = 1
        coordinate: float = 0.0
        hsp_str: str = 'Label       Index       Coordinate\n'
        hsp_str += get_hsp_row_str(flatten_paths[0], index, coordinate)
        for ii in range(1, len(flatten_paths)):
            if [flatten_paths[ii - 1], flatten_paths[ii]] not in discontinue_pairs:
                coordinate += calc_distance(flatten_paths[ii - 1], flatten_paths[ii]) / (2 * np.pi)
                index += int(np.ceil(calc_distance(flatten_paths[ii - 1], flatten_paths[ii]) / self.density + 1))
            else:
                coordinate += 0.0
                index += 1
            hsp_str += get_hsp_row_str(flatten_paths[ii], index, coordinate)
        return hsp_str

    def write_file(self, filename: PathLike):
        """Write HighSymmetryPoint to a file."""
        with zopen(filename, 'wt') as file:
            file.write(self.get_str())