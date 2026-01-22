from typing import Tuple
from cirq import ops, circuits, transformers
from cirq.contrib.paulistring.clifford_target_gateset import CliffordTargetGateset
def _iter_ops_range_reversed(moment_end):
    for i in reversed(range(moment_end)):
        moment = circuit[i]
        for op in moment.operations:
            if not isinstance(op, ops.PauliStringPhasor):
                yield op