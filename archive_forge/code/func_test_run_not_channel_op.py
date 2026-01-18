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
def test_run_not_channel_op(dtype: Type[np.complexfloating], split: bool):

    class BadOp(cirq.Operation):

        def __init__(self, qubits):
            self._qubits = qubits

        @property
        def qubits(self):
            return self._qubits

        def with_qubits(self, *new_qubits):
            return BadOp(self._qubits)
    q0 = cirq.LineQubit(0)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype, split_untangled_states=split)
    circuit = cirq.Circuit([BadOp([q0])])
    with pytest.raises(TypeError):
        simulator.simulate(circuit)