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
class KabschMatcher(MSONable):
    """Molecule matcher using Kabsch algorithm.

    The Kabsch algorithm capable aligning two molecules by finding the parameters
    (translation, rotation) which minimize the root-mean-square-deviation (RMSD) of
    two molecules which are topologically (atom types, geometry) similar two each other.

    Notes:
        When aligning molecules, the atoms of the two molecules **must** be in the same
        order for the results to be sensible.
    """

    def __init__(self, target: Molecule):
        """Constructor of the matcher object.

        Args:
            target: a `Molecule` object used as a target during the alignment
        """
        self.target = target

    def match(self, p: Molecule):
        """Using the Kabsch algorithm the alignment of two molecules (P, Q)
        happens in three steps:
        - translate the P and Q into their centroid
        - compute of the optimal rotation matrix (U) using Kabsch algorithm
        - compute the translation (V) and rmsd.

        The function returns the rotation matrix (U), translation vector (V),
        and RMSD between Q and P', where P' is:

            P' = P * U + V

        Args:
            p: a `Molecule` object what will be matched with the target one.

        Returns:
            U: Rotation matrix (D,D)
            V: Translation vector (D)
            RMSD : Root mean squared deviation between P and Q
        """
        if self.target.atomic_numbers != p.atomic_numbers:
            raise ValueError("The order of the species aren't matching! Please try using PermInvMatcher.")
        p_coord, q_coord = (p.cart_coords, self.target.cart_coords)
        p_trans, q_trans = (p_coord.mean(axis=0), q_coord.mean(axis=0))
        p_centroid, q_centroid = (p_coord - p_trans, q_coord - q_trans)
        U = self.kabsch(p_centroid, q_centroid)
        p_prime_centroid = np.dot(p_centroid, U)
        rmsd = np.sqrt(np.mean(np.square(p_prime_centroid - q_centroid)))
        V = q_trans - np.dot(p_trans, U)
        return (U, V, rmsd)

    def fit(self, p: Molecule):
        """Rotate and transform `p` molecule according to the best match.

        Args:
            p: a `Molecule` object what will be matched with the target one.

        Returns:
            p_prime: Rotated and translated of the `p` `Molecule` object
            rmsd: Root-mean-square-deviation between `p_prime` and the `target`
        """
        U, V, rmsd = self.match(p)
        p_prime = p.copy()
        for site in p_prime:
            site.coords = np.dot(site.coords, U) + V
        return (p_prime, rmsd)

    @staticmethod
    def kabsch(P: np.ndarray, Q: np.ndarray):
        """The Kabsch algorithm is a method for calculating the optimal rotation matrix
        that minimizes the root mean squared deviation (RMSD) between two paired sets of points
        P and Q, centered around the their centroid.

        For more info see:
        - http://wikipedia.org/wiki/Kabsch_algorithm and
        - https://cnx.org/contents/HV-RsdwL@23/Molecular-Distance-Measures

        Args:
            P: Nx3 matrix, where N is the number of points.
            Q: Nx3 matrix, where N is the number of points.

        Returns:
            U: 3x3 rotation matrix
        """
        C = np.dot(P.T, Q)
        V, _S, WT = np.linalg.svd(C)
        det = np.linalg.det(np.dot(V, WT))
        return np.dot(np.dot(V, np.diag([1, 1, det])), WT)