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
def _generate_G_points(self, kpoint: np.ndarray, gamma: bool=False) -> tuple[list, list, list]:
    """
        Helper function to generate G-points based on nbmax.

        This function iterates over possible G-point values and determines
        if the energy is less than G_{cut}. Valid values are appended to
        the output array. This function should not be called outside of
        initialization.

        Args:
            kpoint (np.array): the array containing the current k-point value
            gamma (bool): determines if G points for gamma-point only executable
                          should be generated

        Returns:
            a list containing valid G-points
        """
    kmax = self._nbmax[0] + 1 if gamma else 2 * self._nbmax[0] + 1
    gpoints = []
    extra_gpoints = []
    extra_coeff_inds = []
    G_ind = 0
    for i in range(2 * self._nbmax[2] + 1):
        i3 = i - 2 * self._nbmax[2] - 1 if i > self._nbmax[2] else i
        for j in range(2 * self._nbmax[1] + 1):
            j2 = j - 2 * self._nbmax[1] - 1 if j > self._nbmax[1] else j
            for k in range(kmax):
                k1 = k - 2 * self._nbmax[0] - 1 if k > self._nbmax[0] else k
                if gamma and (k1 == 0 and j2 < 0 or (k1 == 0 and j2 == 0 and (i3 < 0))):
                    continue
                G = np.array([k1, j2, i3])
                v = kpoint + G
                g = np.linalg.norm(np.dot(v, self.b))
                E = g ** 2 / self._C
                if self.encut > E:
                    gpoints.append(G)
                    if gamma and (k1, j2, i3) != (0, 0, 0):
                        extra_gpoints.append(-G)
                        extra_coeff_inds.append(G_ind)
                    G_ind += 1
    return (gpoints, extra_gpoints, extra_coeff_inds)