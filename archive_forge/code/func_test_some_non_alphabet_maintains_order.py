import numpy as np
import pytest
from opt_einsum import contract, contract_expression, contract_path, helpers
from opt_einsum.paths import linear_to_ssa, ssa_to_linear
def test_some_non_alphabet_maintains_order():
    string = 'c' + chr(ord('b') + 848) + 'a'
    x = np.random.rand(2, 3, 4)
    assert np.allclose(contract(string, x), contract('cxa', x))