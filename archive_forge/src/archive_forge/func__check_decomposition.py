from string import ascii_lowercase
import copy
import pickle
import numpy as np
import pennylane as qml
from pennylane.operation import EigvalsUndefinedError
def _check_decomposition(op):
    """Checks involving the decomposition."""
    if op.has_decomposition:
        decomp = op.decomposition()
        try:
            compute_decomp = type(op).compute_decomposition(*op.data, wires=op.wires, **op.hyperparameters)
        except (qml.operation.DecompositionUndefinedError, TypeError):
            compute_decomp = decomp
        with qml.queuing.AnnotatedQueue() as queued_decomp:
            op.decomposition()
        processed_queue = qml.tape.QuantumTape.from_queue(queued_decomp)
        expand = op.expand()
        assert isinstance(decomp, list), 'decomposition must be a list'
        assert isinstance(compute_decomp, list), 'decomposition must be a list'
        assert isinstance(expand, qml.tape.QuantumScript), 'expand must return a QuantumScript'
        for o1, o2, o3, o4 in zip(decomp, compute_decomp, processed_queue, expand):
            assert o1 == o2, 'decomposition must match compute_decomposition'
            assert o1 == o3, 'decomposition must match queued operations'
            assert o1 == o4, 'decomposition must match expansion'
            assert isinstance(o1, qml.operation.Operator), 'decomposition must contain operators'
    else:
        failure_comment = 'If has_decomposition is False, then decomposition must raise a ``DecompositionUndefinedError``.'
        _assert_error_raised(op.decomposition, qml.operation.DecompositionUndefinedError, failure_comment=failure_comment)()
        _assert_error_raised(op.expand, qml.operation.DecompositionUndefinedError, failure_comment=failure_comment)()
        _assert_error_raised(op.compute_decomposition, qml.operation.DecompositionUndefinedError, failure_comment=failure_comment)(*op.data, wires=op.wires, **op.hyperparameters)