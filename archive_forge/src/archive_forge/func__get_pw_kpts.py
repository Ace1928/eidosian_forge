import re
from collections import OrderedDict
import numpy as np
from ase import Atoms
from ase.units import Hartree, Bohr
from ase.calculators.singlepoint import (SinglePointDFTCalculator,
from .parser import _define_pattern
def _get_pw_kpts(chunk):
    eval_blocks = []
    for block in _nwpw_eval_block.findall(chunk):
        if 'pathlength' not in block:
            eval_blocks.append(block)
    if not eval_blocks:
        return []
    if 'virtual' in eval_blocks[-1]:
        occ_block = eval_blocks[-2]
        virt_block = eval_blocks[-1]
    else:
        occ_block = eval_blocks[-1]
        virt_block = ''
    kpts = NWChemKpts()
    _extract_pw_kpts(occ_block, kpts, 1.0)
    _extract_pw_kpts(virt_block, kpts, 0.0)
    for match in _kpt_weight.finditer(occ_block):
        index, weight = match.groups()
        kpts.set_weight(index, float(weight))
    return (kpts.to_ibz_kpts(), kpts.to_singlepointkpts())