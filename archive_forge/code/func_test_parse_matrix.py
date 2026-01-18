import numpy as np
import pytest
import sympy
from cirq.interop.quirk.cells.parse import parse_matrix, parse_formula, parse_complex
def test_parse_matrix():
    s = np.sqrt(0.5)
    np.testing.assert_allclose(parse_matrix('{{√½,√½},{-√½,√½}}'), np.array([[s, s], [-s, s]]), atol=1e-08)
    np.testing.assert_allclose(parse_matrix('{{√½,√½i},{√½i,√½}}'), np.array([[s, s * 1j], [s * 1j, s]]), atol=1e-08)
    np.testing.assert_allclose(parse_matrix('{{1,-i},{i,1+i}}'), np.array([[1, -1j], [1j, 1 + 1j]]), atol=1e-08)