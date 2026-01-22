from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.util.due import Doi, due
from pymatgen.util.numba import njit
@dataclass
class ZSLMatch(MSONable):
    """
    A match from the Zur and McGill Algorithm. The super_lattice vectors are listed
    as _sl_vectors. These are reduced according to the algorithm in the paper which
    effectively a rotation in 3D space. Use the match_transformation property to get
    the appropriate transformation matrix.
    """
    film_sl_vectors: list
    substrate_sl_vectors: list
    film_vectors: list
    substrate_vectors: list
    film_transformation: list
    substrate_transformation: list

    @property
    def match_area(self):
        """The area of the match between the substrate and film super lattice vectors."""
        return vec_area(*self.film_sl_vectors)

    @property
    def match_transformation(self):
        """The transformation matrix to convert the film super lattice vectors to the substrate."""
        film_matrix = list(self.film_sl_vectors)
        film_matrix.append(np.cross(film_matrix[0], film_matrix[1]))
        film_matrix = np.array(film_matrix, dtype=float)
        substrate_matrix = list(self.substrate_sl_vectors)
        temp_sub = np.cross(substrate_matrix[0], substrate_matrix[1]).astype(float)
        temp_sub = temp_sub * fast_norm(film_matrix[2]) / fast_norm(temp_sub)
        substrate_matrix.append(temp_sub)
        return np.transpose(np.linalg.solve(film_matrix, substrate_matrix))