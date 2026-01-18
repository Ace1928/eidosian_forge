from fractions import Fraction
from decimal import Decimal
from numbers import Number
import numpy as np
import pytest
import sympy
import cirq
def test_numpy_dtype_compatibility():
    i_a, i_b, i_c = (0, 1, 2)
    i_types = [np.intc, np.intp, np.int0, np.int8, np.int16, np.int32, np.int64]
    for i_type in i_types:
        assert cirq.approx_eq(i_type(i_a), i_type(i_b), atol=1)
        assert not cirq.approx_eq(i_type(i_a), i_type(i_c), atol=1)
    u_types = [np.uint, np.uint0, np.uint8, np.uint16, np.uint32, np.uint64]
    for u_type in u_types:
        assert cirq.approx_eq(u_type(i_a), u_type(i_b), atol=1)
        assert not cirq.approx_eq(u_type(i_a), u_type(i_c), atol=1)
    f_a, f_b, f_c = (0, 1e-08, 1)
    f_types = [np.float16, np.float32, np.float64]
    if hasattr(np, 'float128'):
        f_types.append(np.float128)
    for f_type in f_types:
        assert cirq.approx_eq(f_type(f_a), f_type(f_b), atol=1e-08)
        assert not cirq.approx_eq(f_type(f_a), f_type(f_c), atol=1e-08)
    c_a, c_b, c_c = (0, 1e-08j, 1j)
    c_types = [np.complex64, np.complex128]
    if hasattr(np, 'complex256'):
        c_types.append(np.complex256)
    for c_type in c_types:
        assert cirq.approx_eq(c_type(c_a), c_type(c_b), atol=1e-08)
        assert not cirq.approx_eq(c_type(c_a), c_type(c_c), atol=1e-08)