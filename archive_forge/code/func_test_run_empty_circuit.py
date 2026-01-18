import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('split', [True, False])
def test_run_empty_circuit(dtype: Type[np.complexfloating], split: bool):
    simulator = cirq.DensityMatrixSimulator(dtype=dtype, split_untangled_states=split)
    with pytest.raises(ValueError, match='no measurements'):
        simulator.run(cirq.Circuit())