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
def _is_molecule_linear(self, mol):
    """
        Is the molecule a linear one.

        Args:
            mol: The molecule. OpenBabel OBMol object.

        Returns:
            Boolean value.
        """
    if mol.NumAtoms() < 3:
        return True
    a1 = mol.GetAtom(1)
    a2 = mol.GetAtom(2)
    for idx in range(3, mol.NumAtoms() + 1):
        angle = float(mol.GetAtom(idx).GetAngle(a2, a1))
        if angle < 0.0:
            angle = -angle
        if angle > 90.0:
            angle = 180.0 - angle
        if angle > self._angle_tolerance:
            return False
    return True