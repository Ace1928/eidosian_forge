import numpy as np
import pytest
import sympy
from cirq.interop.quirk.cells.parse import parse_matrix, parse_formula, parse_complex
def test_parse_complex_raw_cases_from_quirk():
    assert parse_complex('0') == 0
    assert parse_complex('1') == 1
    assert parse_complex('-1') == -1
    assert parse_complex('i') == 1j
    assert parse_complex('-i') == -1j
    assert parse_complex('2') == 2
    assert parse_complex('2i') == 2j
    assert parse_complex('-2i') == -2j
    assert parse_complex('3-2i') == 3 - 2j
    assert parse_complex('1-i') == 1 - 1j
    assert parse_complex('1+i') == 1 + 1j
    assert parse_complex('-5+2i') == -5 + 2j
    assert parse_complex('-5-2i') == -5 - 2j
    assert parse_complex('3/2i') == 1.5j
    assert parse_complex('√2-⅓i') == np.sqrt(2) - 1j / 3
    assert parse_complex('1e-10') == 1e-10
    assert parse_complex('1e+10') == 10000000000
    assert parse_complex('2.5e-10') == 2.5e-10
    assert parse_complex('2.5E-10') == 2.5e-10
    assert parse_complex('2.5e+10') == 25000000000
    assert parse_complex('2.e+10') == 20000000000
    np.testing.assert_allclose(parse_complex('e'), np.e)
    np.testing.assert_allclose(parse_complex('pi e'), np.pi * np.e)
    np.testing.assert_allclose(parse_complex('pi e 2'), np.pi * np.e * 2)
    np.testing.assert_allclose(parse_complex('2       pi'), 2 * np.pi)