from __future__ import annotations
import os
import re
import warnings
from string import Template
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.analysis.excitation import ExcitationSpectrum
from pymatgen.core.structure import Molecule, Structure
from pymatgen.core.units import Energy, FloatWithUnit
@staticmethod
def _parse_preamble(preamble):
    info = {}
    for line in preamble.split('\n'):
        tokens = line.split('=')
        if len(tokens) > 1:
            info[tokens[0].strip()] = tokens[-1].strip()
    return info