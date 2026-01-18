from symengine import symbols, init_printing
from symengine.lib.symengine_wrapper import (DenseMatrix, Symbol, Integer,
from symengine.test_utilities import raises
import unittest
def test_row_swap():
    A = DenseMatrix(2, 2, [1, 2, 3, 4])
    B = DenseMatrix(2, 2, [3, 4, 1, 2])
    A.row_swap(0, 1)
    assert A == B