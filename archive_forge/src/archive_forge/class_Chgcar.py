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
class Chgcar(VolumetricData):
    """Simple object for reading a CHGCAR file."""

    def __init__(self, poscar, data, data_aug=None):
        """
        Args:
            poscar (Poscar | Structure): Object containing structure.
            data: Actual data.
            data_aug: Augmentation charge data.
        """
        if isinstance(poscar, Poscar):
            struct = poscar.structure
            self.poscar = poscar
            self.name = poscar.comment
        elif isinstance(poscar, Structure):
            struct = poscar
            self.poscar = Poscar(poscar)
            self.name = None
        super().__init__(struct, data, data_aug=data_aug)
        self._distance_matrix = {}

    @classmethod
    def from_file(cls, filename: str) -> Self:
        """Read a CHGCAR file.

        Args:
            filename (str): Path to CHGCAR file.

        Returns:
            Chgcar
        """
        poscar, data, data_aug = VolumetricData.parse_file(filename)
        return cls(poscar, data, data_aug=data_aug)

    @property
    def net_magnetization(self):
        """Net magnetization from Chgcar"""
        if self.is_spin_polarized:
            return np.sum(self.data['diff'])
        return None