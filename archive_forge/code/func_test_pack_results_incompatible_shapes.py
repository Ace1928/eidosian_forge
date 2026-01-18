import numpy as np
import pytest
import sympy
import cirq
import cirq_google.api.v1.programs as programs
from cirq_google.api.v1 import operations_pb2
def test_pack_results_incompatible_shapes():

    def bools(*shape):
        return np.zeros(shape, dtype=bool)
    with pytest.raises(ValueError):
        programs.pack_results([('a', bools(10))])
    with pytest.raises(ValueError):
        programs.pack_results([('a', bools(7, 3)), ('b', bools(8, 2))])