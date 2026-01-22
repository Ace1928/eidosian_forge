from __future__ import annotations
import re
import warnings
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as cst
from monty.io import zopen
from scipy.stats import norm
from pymatgen.core import Composition, Element, Molecule
from pymatgen.core.operations import SymmOp
from pymatgen.core.units import Ha_to_eV
from pymatgen.electronic_structure.core import Spin
from pymatgen.util.coord import get_angle
from pymatgen.util.plotting import pretty_plot
def _check_pcm(self, line):
    energy_patt = re.compile('(Dispersion|Cavitation|Repulsion) energy\\s+\\S+\\s+=\\s+(\\S*)')
    total_patt = re.compile('with all non electrostatic terms\\s+\\S+\\s+=\\s+(\\S*)')
    parameter_patt = re.compile('(Eps|Numeral density|RSolv|Eps\\(inf[inity]*\\))\\s+=\\s*(\\S*)')
    if energy_patt.search(line):
        m = energy_patt.search(line)
        self.pcm[f'{m.group(1)} energy'] = float(m.group(2))
    elif total_patt.search(line):
        m = total_patt.search(line)
        self.pcm['Total energy'] = float(m.group(1))
    elif parameter_patt.search(line):
        m = parameter_patt.search(line)
        self.pcm[m.group(1)] = float(m.group(2))