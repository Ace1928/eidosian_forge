import re
from collections import OrderedDict
import numpy as np
from ase import Atoms
from ase.units import Hartree, Bohr
from ase.calculators.singlepoint import (SinglePointDFTCalculator,
from .parser import _define_pattern
def _extract_pw_kpts(chunk, kpts, default_occ):
    for match in _kpt.finditer(chunk):
        point, weight, raw_kpt, orbitals = match.groups()
        index = int(point) - 1
        for line in orbitals.split('\n'):
            tokens = line.strip().split()
            if not tokens:
                continue
            ntokens = len(tokens)
            a_e = float(tokens[0]) * Hartree
            if ntokens % 3 == 0:
                a_o = default_occ
            else:
                a_o = float(tokens[3].split('=')[1])
            kpts.add_eval(index, 0, a_e, a_o)
            if ntokens <= 4:
                continue
            if ntokens == 6:
                b_e = float(tokens[3]) * Hartree
                b_o = default_occ
            elif ntokens == 8:
                b_e = float(tokens[4]) * Hartree
                b_o = float(tokens[7].split('=')[1])
            kpts.add_eval(index, 1, b_e, b_o)
        kpts.set_weight(index, float(weight))
        kpts.add_ibz_kpt(index, raw_kpt)