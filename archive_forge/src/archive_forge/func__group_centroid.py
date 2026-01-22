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
@staticmethod
def _group_centroid(mol, ilabels, group_atoms):
    """
        Calculate the centroids of a group atoms indexed by the labels of inchi.

        Args:
            mol: The molecule. OpenBabel OBMol object
            ilabel: inchi label map

        Returns:
            Centroid. Tuple (x, y, z)
        """
    c1x, c1y, c1z = (0.0, 0.0, 0.0)
    for idx in group_atoms:
        orig_idx = ilabels[idx - 1]
        oa1 = mol.GetAtom(orig_idx)
        c1x += float(oa1.x())
        c1y += float(oa1.y())
        c1z += float(oa1.z())
    n_atoms = len(group_atoms)
    c1x /= n_atoms
    c1y /= n_atoms
    c1z /= n_atoms
    return (c1x, c1y, c1z)