import os
import sys
import itertools
import traceback
import textwrap
import subprocess
import pytest
import numpy as np
from numpy import array, single, double, csingle, cdouble, dot, identity, matmul
from numpy.core import swapaxes
from numpy import multiply, atleast_2d, inf, asarray
from numpy import linalg
from numpy.linalg import matrix_power, norm, matrix_rank, multi_dot, LinAlgError
from numpy.linalg.linalg import _multi_dot_matrix_chain_order
from numpy.testing import (
def _make_generalized_cases():
    new_cases = []
    for case in CASES:
        if not isinstance(case.a, np.ndarray):
            continue
        a = np.array([case.a, 2 * case.a, 3 * case.a])
        if case.b is None:
            b = None
        else:
            b = np.array([case.b, 7 * case.b, 6 * case.b])
        new_case = LinalgCase(case.name + '_tile3', a, b, tags=case.tags | {'generalized'})
        new_cases.append(new_case)
        a = np.array([case.a] * 2 * 3).reshape((3, 2) + case.a.shape)
        if case.b is None:
            b = None
        else:
            b = np.array([case.b] * 2 * 3).reshape((3, 2) + case.b.shape)
        new_case = LinalgCase(case.name + '_tile213', a, b, tags=case.tags | {'generalized'})
        new_cases.append(new_case)
    return new_cases