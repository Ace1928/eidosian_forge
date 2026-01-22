import os
import warnings
import shutil
from os.path import join, isfile, islink
from typing import List, Sequence, Tuple
import numpy as np
import ase
from ase.calculators.calculator import kpts2ndarray
from ase.calculators.vasp.setups import get_default_setups
def _make_sort(self, atoms: ase.Atoms, special_setups: Sequence[int]=()) -> Tuple[List[int], List[int]]:
    symbols, _ = count_symbols(atoms, exclude=special_setups)
    srt = []
    srt.extend(special_setups)
    for symbol in symbols:
        for m, atom in enumerate(atoms):
            if m in special_setups:
                continue
            if atom.symbol == symbol:
                srt.append(m)
    resrt = list(range(len(srt)))
    for n in range(len(resrt)):
        resrt[srt[n]] = n
    return (srt, resrt)