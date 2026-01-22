import functools
import itertools
import numbers
import warnings
import numpy as np
from scipy.linalg import solve as linalg_solve
import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.ops.functions import bind_new_parameters
from pennylane.tape import QuantumScript
def _copy_and_shift_params(tape, indices, shifts, multipliers, cast=False):
    """Create a copy of a tape and of parameters, and set the new tape to the parameters
    rescaled and shifted as indicated by ``indices``, ``multipliers`` and ``shifts``."""
    all_ops = tape.circuit
    for idx, shift, multiplier in zip(indices, shifts, multipliers):
        _, op_idx, p_idx = tape.get_operation(idx)
        op = all_ops[op_idx].obs if isinstance(all_ops[op_idx], MeasurementProcess) else all_ops[op_idx]
        new_params = list(op.data)
        if not isinstance(new_params[p_idx], numbers.Integral):
            multiplier = qml.math.convert_like(multiplier, new_params[p_idx])
            multiplier = qml.math.cast_like(multiplier, new_params[p_idx])
            shift = qml.math.convert_like(shift, new_params[p_idx])
            shift = qml.math.cast_like(shift, new_params[p_idx])
        new_params[p_idx] = new_params[p_idx] * multiplier
        new_params[p_idx] = new_params[p_idx] + shift
        if cast:
            dtype = getattr(new_params[p_idx], 'dtype', float)
            new_params[p_idx] = qml.math.cast(new_params[p_idx], dtype)
        shifted_op = bind_new_parameters(op, new_params)
        if op_idx < len(tape.operations):
            all_ops[op_idx] = shifted_op
        else:
            mp = all_ops[op_idx].__class__
            all_ops[op_idx] = mp(obs=shifted_op)
    ops = all_ops[:len(tape.operations)]
    meas = all_ops[len(tape.operations):]
    return QuantumScript(ops=ops, measurements=meas, shots=tape.shots)