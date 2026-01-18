from symengine import symbols, init_printing
from symengine.lib.symengine_wrapper import (DenseMatrix, Symbol, Integer,
from symengine.test_utilities import raises
import unittest
def test_rowadd():
    M = ones(3)
    assert M.rowadd(2, 1, 1) == DenseMatrix([[1, 1, 1], [1, 1, 1], [2, 2, 2]])