from symengine import symbols, init_printing
from symengine.lib.symengine_wrapper import (DenseMatrix, Symbol, Integer,
from symengine.test_utilities import raises
import unittest
def test_rowmul():
    M = ones(3)
    assert M.rowmul(2, 2) == DenseMatrix([[1, 1, 1], [1, 1, 1], [2, 2, 2]])