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
@staticmethod
def _parse_dos(elem):
    efermi = float(elem.find('i').text)
    energies = None
    tdensities = {}
    idensities = {}
    for s in elem.find('total').find('array').find('set').findall('set'):
        data = np.array(_parse_vasp_array(s))
        energies = data[:, 0]
        spin = Spin.up if s.attrib['comment'] == 'spin 1' else Spin.down
        tdensities[spin] = data[:, 1]
        idensities[spin] = data[:, 2]
    pdoss = []
    partial = elem.find('partial')
    if partial is not None:
        orbs = [ss.text for ss in partial.find('array').findall('field')]
        orbs.pop(0)
        lm = any(('x' in s for s in orbs))
        for s in partial.find('array').find('set').findall('set'):
            pdos = defaultdict(dict)
            for ss in s.findall('set'):
                spin = Spin.up if ss.attrib['comment'] == 'spin 1' else Spin.down
                data = np.array(_parse_vasp_array(ss))
                _nrow, ncol = data.shape
                for j in range(1, ncol):
                    orb = Orbital(j - 1) if lm else OrbitalType(j - 1)
                    pdos[orb][spin] = data[:, j]
            pdoss.append(pdos)
    elem.clear()
    return (Dos(efermi, energies, tdensities), Dos(efermi, energies, idensities), pdoss)