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
class Elfcar(VolumetricData):
    """
    Read an ELFCAR file which contains the Electron Localization Function (ELF)
    as calculated by VASP.

    For ELF, "total" key refers to Spin.up, and "diff" refers to Spin.down.

    This also contains information on the kinetic energy density.
    """

    def __init__(self, poscar, data):
        """
        Args:
            poscar (Poscar or Structure): Object containing structure.
            data: Actual data.
        """
        if isinstance(poscar, Poscar):
            tmp_struct = poscar.structure
            self.poscar = poscar
        elif isinstance(poscar, Structure):
            tmp_struct = poscar
            self.poscar = Poscar(poscar)
        super().__init__(tmp_struct, data)
        self.data = data

    @classmethod
    def from_file(cls, filename: str) -> Self:
        """
        Reads a ELFCAR file.

        Args:
            filename: Filename

        Returns:
            Elfcar
        """
        poscar, data, _data_aug = VolumetricData.parse_file(filename)
        return cls(poscar, data)

    def get_alpha(self):
        """Get the parameter alpha where ELF = 1/(1+alpha^2)."""
        alpha_data = {}
        for key, val in self.data.items():
            alpha = 1 / val
            alpha = alpha - 1
            alpha = np.sqrt(alpha)
            alpha_data[key] = alpha
        return VolumetricData(self.structure, alpha_data)