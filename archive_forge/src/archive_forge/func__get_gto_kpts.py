import re
from collections import OrderedDict
import numpy as np
from ase import Atoms
from ase.units import Hartree, Bohr
from ase.calculators.singlepoint import (SinglePointDFTCalculator,
from .parser import _define_pattern
def _get_gto_kpts(chunk):
    eval_blocks = _eval_block.findall(chunk)
    if not eval_blocks:
        return []
    kpts = []
    kpt = _get_gto_evals(eval_blocks[-1])
    if kpt.s == 1:
        kpts.append(_get_gto_evals(eval_blocks[-2]))
    kpts.append(kpt)
    return kpts