from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
def test_tagged_act_on():

    class YesActOn(cirq.Gate):

        def _num_qubits_(self) -> int:
            return 1

        def _act_on_(self, sim_state, qubits):
            return True

    class NoActOn(cirq.Gate):

        def _num_qubits_(self) -> int:
            return 1

        def _act_on_(self, sim_state, qubits):
            return NotImplemented

    class MissingActOn(cirq.Operation):

        def with_qubits(self, *new_qubits):
            raise NotImplementedError()

        @property
        def qubits(self):
            pass
    q = cirq.LineQubit(1)
    from cirq.protocols.act_on_protocol_test import ExampleSimulationState
    args = ExampleSimulationState()
    cirq.act_on(YesActOn()(q).with_tags('test'), args)
    with pytest.raises(TypeError, match='Failed to act'):
        cirq.act_on(NoActOn()(q).with_tags('test'), args)
    with pytest.raises(TypeError, match='Failed to act'):
        cirq.act_on(MissingActOn().with_tags('test'), args)