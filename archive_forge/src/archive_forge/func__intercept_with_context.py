import itertools
from typing import Optional
from unittest import mock
import pytest
import cirq
def _intercept_with_context(op: cirq.Operation, context: Optional[cirq.DecompositionContext]=None):
    assert context is not None
    if op.gate == cirq.SWAP:
        q = context.qubit_manager.qalloc(1)
        a, b = op.qubits
        return [cirq.X(a), cirq.X(*q), cirq.X(b)]
    return NotImplemented