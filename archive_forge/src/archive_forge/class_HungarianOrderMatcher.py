from __future__ import annotations
import abc
import copy
import itertools
import logging
import math
import re
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.json import MSONable
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from pymatgen.core.structure import Molecule
class HungarianOrderMatcher(KabschMatcher):
    """This method pre-aligns the molecules based on their principal inertia
    axis and then re-orders the input atom list using the Hungarian method.

    Notes:
        This method cannot guarantee the best match but is very fast.

        When aligning molecules, the atoms of the two molecules **must** have same number
        of atoms from the same species.
    """

    def match(self, p: Molecule):
        """Similar as `KabschMatcher.match` but this method also finds the order of
        atoms which belongs to the best match.

        Args:
            p: a `Molecule` object what will be matched with the target one.

        Returns:
            inds: The indices of atoms
            U: 3x3 rotation matrix
            V: Translation vector
            rmsd: Root mean squared deviation between P and Q
        """
        if sorted(p.atomic_numbers) != sorted(self.target.atomic_numbers):
            raise ValueError("The number of the same species aren't matching!")
        p_coord, q_coord = (p.cart_coords, self.target.cart_coords)
        p_atoms, q_atoms = (np.array(p.atomic_numbers), np.array(self.target.atomic_numbers))
        p_weights = np.array([site.species.weight for site in p])
        q_weights = np.array([site.species.weight for site in self.target])
        p_trans, q_trans = (p.center_of_mass, self.target.center_of_mass)
        p_centroid, q_centroid = (p_coord - p_trans, q_coord - q_trans)
        rmsd = np.inf
        for p_inds_test in self.permutations(p_atoms, p_centroid, p_weights, q_atoms, q_centroid, q_weights):
            p_centroid_test = p_centroid[p_inds_test]
            U_test = self.kabsch(p_centroid_test, q_centroid)
            p_centroid_prime_test = np.dot(p_centroid_test, U_test)
            rmsd_test = np.sqrt(np.mean(np.square(p_centroid_prime_test - q_centroid)))
            if rmsd_test < rmsd:
                inds, U, rmsd = (p_inds_test, U_test, rmsd_test)
        V = q_trans - np.dot(p_trans, U)
        return (inds, U, V, rmsd)

    def fit(self, p: Molecule):
        """Order, rotate and transform `p` molecule according to the best match.

        Args:
            p: a `Molecule` object what will be matched with the target one.

        Returns:
            p_prime: Rotated and translated of the `p` `Molecule` object
            rmsd: Root-mean-square-deviation between `p_prime` and the `target`
        """
        inds, U, V, rmsd = self.match(p)
        p_prime = Molecule.from_sites([p[idx] for idx in inds])
        for site in p_prime:
            site.coords = np.dot(site.coords, U) + V
        return (p_prime, rmsd)

    @staticmethod
    def permutations(p_atoms, p_centroid, p_weights, q_atoms, q_centroid, q_weights):
        """Generates two possible permutations of atom order. This method uses the principle component
        of the inertia tensor to pre-align the molecules and hungarian method to determine the order.
        There are always two possible permutation depending on the way to pre-aligning the molecules.

        Args:
            p_atoms: atom numbers
            p_centroid: array of atom positions
            p_weights: array of atom weights
            q_atoms: atom numbers
            q_centroid: array of atom positions
            q_weights: array of atom weights

        Yield:
            perm_inds: array of atoms' order
        """
        p_axis = HungarianOrderMatcher.get_principal_axis(p_centroid, p_weights)
        q_axis = HungarianOrderMatcher.get_principal_axis(q_centroid, q_weights)
        U = HungarianOrderMatcher.rotation_matrix_vectors(q_axis, p_axis)
        p_centroid_test = np.dot(p_centroid, U)
        perm_inds = np.zeros(len(p_atoms), dtype=int)
        species = np.unique(p_atoms)
        for specie in species:
            p_atom_inds = np.where(p_atoms == specie)[0]
            q_atom_inds = np.where(q_atoms == specie)[0]
            A = q_centroid[q_atom_inds]
            B = p_centroid_test[p_atom_inds]
            distances = cdist(A, B, 'euclidean')
            _a_inds, b_inds = linear_sum_assignment(distances)
            perm_inds[q_atom_inds] = p_atom_inds[b_inds]
        yield perm_inds
        U = HungarianOrderMatcher.rotation_matrix_vectors(q_axis, -p_axis)
        p_centroid_test = np.dot(p_centroid, U)
        perm_inds = np.zeros(len(p_atoms), dtype=int)
        species = np.unique(p_atoms)
        for specie in species:
            p_atom_inds = np.where(p_atoms == specie)[0]
            q_atom_inds = np.where(q_atoms == specie)[0]
            A = q_centroid[q_atom_inds]
            B = p_centroid_test[p_atom_inds]
            distances = cdist(A, B, 'euclidean')
            _a_inds, b_inds = linear_sum_assignment(distances)
            perm_inds[q_atom_inds] = p_atom_inds[b_inds]
        yield perm_inds

    @staticmethod
    def get_principal_axis(coords, weights):
        """Get the molecule's principal axis.

        Args:
            coords: coordinates of atoms
            weights: the weight use for calculating the inertia tensor

        Returns:
            Array of dim 3 containing the principal axis
        """
        Ixx = Iyy = Izz = Ixy = Ixz = Iyz = 0.0
        for (x, y, z), wt in zip(coords, weights):
            Ixx += wt * (y * y + z * z)
            Iyy += wt * (x * x + z * z)
            Izz += wt * (x * x + y * y)
            Ixy += -wt * x * y
            Ixz += -wt * x * z
            Iyz += -wt * y * z
        inertia_tensor = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
        _eigvals, eigvecs = np.linalg.eigh(inertia_tensor)
        return eigvecs[:, 0]

    @staticmethod
    def rotation_matrix_vectors(v1, v2):
        """Returns the rotation matrix that rotates v1 onto v2 using
        Rodrigues' rotation formula.

        See more: https://math.stackexchange.com/a/476311

        Args:
            v1: initial vector
            v2: target vector

        Returns:
            3x3 rotation matrix
        """
        if np.allclose(v1, v2):
            return np.eye(3)
        if np.allclose(v1, -v2):
            return np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        v = np.cross(v1, v2)
        norm = np.linalg.norm(v)
        c = np.vdot(v1, v2)
        vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
        return np.eye(3) + vx + np.dot(vx, vx) * ((1.0 - c) / (norm * norm))