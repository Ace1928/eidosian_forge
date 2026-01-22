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
def _generate_nbmax(self) -> None:
    """
        Helper function that determines maximum number of b vectors for
        each direction.

        This algorithm is adapted from WaveTrans (see Class docstring). There
        should be no reason for this function to be called outside of
        initialization.
        """
    bmag = np.linalg.norm(self.b, axis=1)
    b = self.b
    phi12 = np.arccos(np.dot(b[0, :], b[1, :]) / (bmag[0] * bmag[1]))
    sphi123 = np.dot(b[2, :], np.cross(b[0, :], b[1, :])) / (bmag[2] * np.linalg.norm(np.cross(b[0, :], b[1, :])))
    nbmaxA = np.sqrt(self.encut * self._C) / bmag
    nbmaxA[0] /= np.abs(np.sin(phi12))
    nbmaxA[1] /= np.abs(np.sin(phi12))
    nbmaxA[2] /= np.abs(sphi123)
    nbmaxA += 1
    phi13 = np.arccos(np.dot(b[0, :], b[2, :]) / (bmag[0] * bmag[2]))
    sphi123 = np.dot(b[1, :], np.cross(b[0, :], b[2, :])) / (bmag[1] * np.linalg.norm(np.cross(b[0, :], b[2, :])))
    nbmaxB = np.sqrt(self.encut * self._C) / bmag
    nbmaxB[0] /= np.abs(np.sin(phi13))
    nbmaxB[1] /= np.abs(sphi123)
    nbmaxB[2] /= np.abs(np.sin(phi13))
    nbmaxB += 1
    phi23 = np.arccos(np.dot(b[1, :], b[2, :]) / (bmag[1] * bmag[2]))
    sphi123 = np.dot(b[0, :], np.cross(b[1, :], b[2, :])) / (bmag[0] * np.linalg.norm(np.cross(b[1, :], b[2, :])))
    nbmaxC = np.sqrt(self.encut * self._C) / bmag
    nbmaxC[0] /= np.abs(sphi123)
    nbmaxC[1] /= np.abs(np.sin(phi23))
    nbmaxC[2] /= np.abs(np.sin(phi23))
    nbmaxC += 1
    self._nbmax = np.max([nbmaxA, nbmaxB, nbmaxC], axis=0).astype(int)