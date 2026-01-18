from __future__ import annotations
from sympy.ntheory import qs
from sympy.ntheory.qs import SievePolynomial, _generate_factor_base, \
from sympy.testing.pytest import slow
def test_qs_3():
    N = 1817
    smooth_relations = [(2455024, 637, [0, 0, 0, 1]), (-27993000, 81536, [0, 1, 0, 1]), (11461840, 12544, [0, 0, 0, 0]), (149, 20384, [0, 1, 0, 1]), (-31138074, 19208, [0, 1, 0, 0])]
    matrix = _build_matrix(smooth_relations)
    assert matrix == [[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 0, 0], [0, 1, 0, 1], [0, 1, 0, 0]]
    dependent_row, mark, gauss_matrix = _gauss_mod_2(matrix)
    assert dependent_row == [[[0, 0, 0, 0], 2], [[0, 1, 0, 0], 3]]
    assert mark == [True, True, False, False, True]
    assert gauss_matrix == [[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 1]]
    factor = _find_factor(dependent_row, mark, gauss_matrix, 0, smooth_relations, N)
    assert factor == 23