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
Generates all of possible permutations of atom order according the threshold.

        Args:
            p: a `Molecule` object what will be matched with the target one.

        Returns:
            Array of index arrays
        