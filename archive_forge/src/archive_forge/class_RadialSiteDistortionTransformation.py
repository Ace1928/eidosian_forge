from __future__ import annotations
import itertools
import logging
import math
import time
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.analysis.ewald import EwaldMinimizer, EwaldSummation
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.transformation_abc import AbstractTransformation
class RadialSiteDistortionTransformation(AbstractTransformation):
    """Radially perturbs atoms around a site. Can be used to create spherical distortion due to a
    point defect.
    """

    def __init__(self, site_index: int, displacement: float=0.1, nn_only: bool=False) -> None:
        """
        Args:
            site_index (int): index of the site in structure to place at the center of the distortion (will
                not be distorted). This index must be provided before the structure is provided in
                apply_transformation in order to keep in line with the base class.
            displacement (float): distance to perturb the atoms around the objective site
            nn_only (bool): Whether to perturb beyond the nearest neighbors. If True, then only the
                nearest neighbors will be perturbed, leaving the other sites undisturbed. If False, then
                the nearest neighbors will receive the full displacement, and then subsequent sites will receive
                a displacement=0.1 / r, where r is the distance each site to the origin site. For small displacements,
                atoms beyond the NN environment will receive very small displacements, and these are almost equal.
                For large displacements, this difference is noticeable.
        """
        self.site_index = site_index
        self.displacement = displacement
        self.nn_only = nn_only

    def apply_transformation(self, structure: Structure):
        """Apply the transformation.

        Args:
            structure: Structure or Molecule to apply the transformation to

        Returns:
            the transformed structure
        """
        structure = structure.copy()
        site = structure[self.site_index]

        def displace_dist(x, r, r0):
            return x * r0 / r
        r0 = max((site.distance(_['site']) for _ in MinimumDistanceNN().get_nn_info(structure, self.site_index)))
        if hasattr(structure, 'lattice'):
            latt_mat = structure.lattice.matrix
            latt_mat = (abs(latt_mat) > 1e-05) * latt_mat
            a, b, c = (latt_mat[0], latt_mat[1], latt_mat[2])
            x = abs(np.dot(a, np.cross(b, c)) / np.linalg.norm(np.cross(b, c)))
            y = abs(np.dot(b, np.cross(a, c)) / np.linalg.norm(np.cross(a, c)))
            z = abs(np.dot(c, np.cross(a, b)) / np.linalg.norm(np.cross(a, b)))
            r_max = np.floor(min([x, y, z]) / 2)
        else:
            r_max = np.max(structure.distance_matrix)
        for vals in structure.get_neighbors(site, r=r0 if self.nn_only else r_max):
            site2, distance, index = vals[:3]
            vec = site2.coords - site.coords
            kwargs = {'indices': [index], 'vector': vec * displace_dist(self.displacement, distance, r0) / np.linalg.norm(vec)}
            if hasattr(structure, 'lattice'):
                kwargs['frac_coords'] = False
            structure.translate_sites(**kwargs)
        return structure

    @property
    def inverse(self):
        """Returns the inverse transformation if available.
        Otherwise, should return None.
        """
        return False

    @property
    def is_one_to_many(self) -> bool:
        """Determines if a Transformation is a one-to-many transformation. If a
        Transformation is a one-to-many transformation, the
        apply_transformation method should have a keyword arg
        "return_ranked_list" which allows for the transformed structures to be
        returned as a ranked list.
        """
        return False

    @property
    def use_multiprocessing(self):
        """Indicates whether the transformation can be applied by a
        subprocessing pool. This should be overridden to return True for
        transformations that the transmuter can parallelize.
        """
        return False