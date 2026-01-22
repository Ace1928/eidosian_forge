from __future__ import annotations
import datetime
import itertools
import logging
import math
import os
import re
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from glob import glob
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.io import reverse_readfile, zopen
from monty.json import MSONable, jsanitize
from monty.os.path import zpath
from monty.re import regrep
from numpy.testing import assert_allclose
from pymatgen.core import Composition, Element, Lattice, Structure
from pymatgen.core.units import unitized
from pymatgen.electronic_structure.bandstructure import (
from pymatgen.electronic_structure.core import Magmom, Orbital, OrbitalType, Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.io.common import VolumetricData as BaseVolumetricData
from pymatgen.io.core import ParseError
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar
from pymatgen.io.wannier90 import Unk
from pymatgen.util.io_utils import clean_lines, micro_pyawk
from pymatgen.util.num import make_symmetric_matrix_from_upper_tri
class Locpot(VolumetricData):
    """Simple object for reading a LOCPOT file."""

    def __init__(self, poscar: Poscar, data: np.ndarray, **kwargs) -> None:
        """
        Args:
            poscar (Poscar): Poscar object containing structure.
            data (np.ndarray): Actual data.
        """
        super().__init__(poscar.structure, data, **kwargs)
        self.name = poscar.comment

    @classmethod
    def from_file(cls, filename: str, **kwargs) -> Self:
        """Read a LOCPOT file.

        Args:
            filename (str): Path to LOCPOT file.

        Returns:
            Locpot
        """
        poscar, data, _data_aug = VolumetricData.parse_file(filename)
        return cls(poscar, data, **kwargs)