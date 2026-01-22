from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.util.due import Doi, due
from pymatgen.util.numba import njit
@due.dcite(Doi('10.1063/1.333084'), description='Lattice match: An application to heteroepitaxy')
class ZSLGenerator(MSONable):
    """
    This class generate matching interface super lattices based on the methodology
    of lattice vector matching for heterostructural interfaces proposed by
    Zur and McGill:
    Journal of Applied Physics 55 (1984), 378 ; doi: 10.1063/1.333084
    The process of generating all possible matching super lattices is:
    1.) Reduce the surface lattice vectors and calculate area for the surfaces
    2.) Generate all super lattice transformations within a maximum allowed area
        limit that give nearly equal area super-lattices for the two
        surfaces - generate_sl_transformation_sets
    3.) For each superlattice set:
        1.) Reduce super lattice vectors
        2.) Check length and angle between film and substrate super lattice
            vectors to determine if the super lattices are the nearly same
            and therefore coincident - get_equiv_transformations.
    """

    def __init__(self, max_area_ratio_tol=0.09, max_area=400, max_length_tol=0.03, max_angle_tol=0.01, bidirectional=False):
        """
        Initialize a Zur Super Lattice Generator for a specific film and
            substrate

        Args:
            max_area_ratio_tol(float): Max tolerance on ratio of
                super-lattices to consider equal
            max_area(float): max super lattice area to generate in search
            max_length_tol: maximum length tolerance in checking if two
                vectors are of nearly the same length
            max_angle_tol: maximum angle tolerance in checking of two sets
                of vectors have nearly the same angle between them.
        """
        self.max_area_ratio_tol = max_area_ratio_tol
        self.max_area = max_area
        self.max_length_tol = max_length_tol
        self.max_angle_tol = max_angle_tol
        self.bidirectional = bidirectional

    def generate_sl_transformation_sets(self, film_area, substrate_area):
        """
        Generates transformation sets for film/substrate pair given the
        area of the unit cell area for the film and substrate. The
        transformation sets map the film and substrate unit cells to super
        lattices with a maximum area

        Args:
            film_area (int): the unit cell area for the film
            substrate_area (int): the unit cell area for the substrate

        Returns:
            transformation_sets: a set of transformation_sets defined as:
                1.) the transformation matrices for the film to create a
                super lattice of area i*film area
                2.) the transformation matrices for the substrate to create
                a super lattice of area j*film area.
        """
        transformation_indices = [(ii, jj) for ii in range(1, int(np.ceil(self.max_area / film_area))) for jj in range(1, int(np.ceil(self.max_area / substrate_area))) if np.absolute(film_area / substrate_area - float(jj) / ii) < self.max_area_ratio_tol] + [(ii, jj) for ii in range(1, int(np.ceil(self.max_area / film_area))) for jj in range(1, int(np.ceil(self.max_area / substrate_area))) if np.absolute(substrate_area / film_area - float(ii) / jj) < self.max_area_ratio_tol]
        transformation_indices = list(set(transformation_indices))
        for ii, jj in sorted(transformation_indices, key=lambda x: x[0] * x[1]):
            yield (gen_sl_transform_matrices(ii), gen_sl_transform_matrices(jj))

    def get_equiv_transformations(self, transformation_sets, film_vectors, substrate_vectors):
        """
        Applies the transformation_sets to the film and substrate vectors
        to generate super-lattices and checks if they matches.
        Returns all matching vectors sets.

        Args:
            transformation_sets(array): an array of transformation sets:
                each transformation set is an array with the (i,j)
                indicating the area multiples of the film and substrate it
                corresponds to, an array with all possible transformations
                for the film area multiple i and another array for the
                substrate area multiple j.
            film_vectors(array): film vectors to generate super lattices
            substrate_vectors(array): substrate vectors to generate super
                lattices
        """
        for film_transformations, substrate_transformations in transformation_sets:
            films = np.array([reduce_vectors(*v) for v in np.dot(film_transformations, film_vectors)], dtype=float)
            substrates = np.array([reduce_vectors(*v) for v in np.dot(substrate_transformations, substrate_vectors)], dtype=float)
            for (f_trans, s_trans), (f, s) in zip(product(film_transformations, substrate_transformations), product(films, substrates)):
                if is_same_vectors(f, s, bidirectional=self.bidirectional, max_length_tol=self.max_length_tol, max_angle_tol=self.max_angle_tol):
                    yield [f, s, f_trans, s_trans]

    def __call__(self, film_vectors, substrate_vectors, lowest=False) -> Iterator[ZSLMatch]:
        """
        Runs the ZSL algorithm to generate all possible matching
        """
        film_area = vec_area(*film_vectors)
        substrate_area = vec_area(*substrate_vectors)
        transformation_sets = self.generate_sl_transformation_sets(film_area, substrate_area)
        equiv_transformations = self.get_equiv_transformations(transformation_sets, film_vectors, substrate_vectors)
        for match in equiv_transformations:
            yield ZSLMatch(film_sl_vectors=match[0], substrate_sl_vectors=match[1], film_vectors=film_vectors, substrate_vectors=substrate_vectors, film_transformation=match[2], substrate_transformation=match[3])
            if lowest:
                break