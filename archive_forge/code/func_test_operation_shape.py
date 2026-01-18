from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
def test_operation_shape():

    class FixedQids(cirq.Operation):

        def with_qubits(self, *new_qids):
            raise NotImplementedError

    class QubitOp(FixedQids):

        @property
        def qubits(self):
            return cirq.LineQubit.range(2)

    class NumQubitOp(FixedQids):

        @property
        def qubits(self):
            return cirq.LineQubit.range(3)

        def _num_qubits_(self):
            return 3

    class ShapeOp(FixedQids):

        @property
        def qubits(self):
            return cirq.LineQubit.range(4)

        def _qid_shape_(self):
            return (1, 2, 3, 4)
    qubit_op = QubitOp()
    assert len(qubit_op.qubits) == 2
    assert cirq.qid_shape(qubit_op) == (2, 2)
    assert cirq.num_qubits(qubit_op) == 2
    num_qubit_op = NumQubitOp()
    assert len(num_qubit_op.qubits) == 3
    assert cirq.qid_shape(num_qubit_op) == (2, 2, 2)
    assert cirq.num_qubits(num_qubit_op) == 3
    shape_op = ShapeOp()
    assert len(shape_op.qubits) == 4
    assert cirq.qid_shape(shape_op) == (1, 2, 3, 4)
    assert cirq.num_qubits(shape_op) == 4