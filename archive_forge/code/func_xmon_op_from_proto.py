import json
from typing import Any, cast, Dict, Optional, Sequence, Tuple, TYPE_CHECKING, Iterator
import numpy as np
import sympy
import cirq
from cirq_google.api.v1 import operations_pb2
def xmon_op_from_proto(proto: operations_pb2.Operation) -> cirq.Operation:
    """Convert the proto to the corresponding operation.

    See protos in api/google/v1 for specification of the protos.

    Args:
        proto: Operation proto.

    Returns:
        The operation.

    Raises:
        ValueError: If the proto has an operation that is invalid.
    """
    param = _parameterized_value_from_proto
    qubit = _qubit_from_proto
    if proto.HasField('exp_w'):
        exp_w = proto.exp_w
        return cirq.PhasedXPowGate(exponent=param(exp_w.half_turns), phase_exponent=param(exp_w.axis_half_turns)).on(qubit(exp_w.target))
    if proto.HasField('exp_z'):
        exp_z = proto.exp_z
        return cirq.Z(qubit(exp_z.target)) ** param(exp_z.half_turns)
    if proto.HasField('exp_11'):
        exp_11 = proto.exp_11
        return cirq.CZ(qubit(exp_11.target1), qubit(exp_11.target2)) ** param(exp_11.half_turns)
    if proto.HasField('measurement'):
        meas = proto.measurement
        return cirq.MeasurementGate(num_qubits=len(meas.targets), key=meas.key, invert_mask=tuple(meas.invert_mask)).on(*[qubit(q) for q in meas.targets])
    raise ValueError(f'invalid operation: {proto}')