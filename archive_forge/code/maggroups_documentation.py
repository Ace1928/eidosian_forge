from __future__ import annotations
import os
import sqlite3
import textwrap
from array import array
from fractions import Fraction
from typing import TYPE_CHECKING
import numpy as np
from monty.design_patterns import cached_class
from pymatgen.core.operations import MagSymmOp
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.groups import SymmetryGroup, in_array_list
from pymatgen.symmetry.settings import JonesFaithfulTransformation
from pymatgen.util.string import transformation_to_string
Parses compact binary representation into transformation between OG and BNS settings.