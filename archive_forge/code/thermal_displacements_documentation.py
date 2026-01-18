from __future__ import annotations
import re
from functools import partial
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifFile, CifParser, CifWriter, str2float
from pymatgen.symmetry.groups import SYMM_DATA
from pymatgen.util.due import Doi, due
Reads a cif with P1 symmetry including positions and ADPs.
        Currently, no check of symmetry is performed as CifParser methods cannot be easily reused.

        Args:
            filename: Filename of the CIF.

        Returns:
            ThermalDisplacementMatrices
        