import os
import sys
import re
import numpy as np
import subprocess
from contextlib import contextmanager
from pathlib import Path
from warnings import warn
from typing import Dict, Any
from xml.etree import ElementTree
import ase
from ase.io import read, jsonio
from ase.utils import PurePath
from ase.calculators import calculator
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.calculators.vasp.create_input import GenerateVaspInput
def _read_magnetic_moment(self, lines=None):
    """Read magnetic moment from OUTCAR"""
    if not lines:
        lines = self.load_file('OUTCAR')
    for n, line in enumerate(lines):
        if 'number of electron  ' in line:
            magnetic_moment = float(line.split()[-1])
    return magnetic_moment