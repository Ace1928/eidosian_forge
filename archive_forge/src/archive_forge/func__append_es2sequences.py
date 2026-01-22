from __future__ import annotations
import operator
from functools import reduce
from math import sqrt
import numpy as np
from scipy.special import erf
def _append_es2sequences(sequences, es):
    result = []
    if not sequences:
        for e in es:
            result.append([e])
    else:
        for e in es:
            result += [[*seq, e] for seq in sequences]
    return result