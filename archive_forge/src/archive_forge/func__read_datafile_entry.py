import os
import warnings
from functools import total_ordering
from typing import Union
import numpy as np
def _read_datafile_entry(spg, no, symbol, setting, f):
    """Read space group data from f to spg."""
    floats = {'0.0': 0.0, '1.0': 1.0, '0': 0.0, '1': 1.0, '-1': -1.0}
    for n, d in [(1, 2), (1, 3), (2, 3), (1, 4), (3, 4), (1, 6), (5, 6)]:
        floats['{0}/{1}'.format(n, d)] = n / d
        floats['-{0}/{1}'.format(n, d)] = -n / d
    spg._no = no
    spg._symbol = symbol.strip()
    spg._setting = setting
    spg._centrosymmetric = bool(int(f.readline().split()[1]))
    f.readline()
    spg._scaled_primitive_cell = np.array([[float(floats.get(s, s)) for s in f.readline().split()] for i in range(3)], dtype=float)
    f.readline()
    spg._reciprocal_cell = np.array([[int(i) for i in f.readline().split()] for i in range(3)], dtype=int)
    spg._nsubtrans = int(f.readline().split()[0])
    spg._subtrans = np.array([[float(floats.get(t, t)) for t in f.readline().split()] for i in range(spg._nsubtrans)], dtype=float)
    nsym = int(f.readline().split()[0])
    symop = np.array([[float(floats.get(s, s)) for s in f.readline().split()] for i in range(nsym)], dtype=float)
    spg._nsymop = nsym
    spg._rotations = np.array(symop[:, :9].reshape((nsym, 3, 3)), dtype=int)
    spg._translations = symop[:, 9:]