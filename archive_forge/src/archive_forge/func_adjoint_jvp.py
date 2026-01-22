from numbers import Number
from typing import Tuple
import numpy as np
import pennylane as qml
from pennylane.operation import operation_derivative
from pennylane.tape import QuantumTape
from .apply_operation import apply_operation
from .simulate import get_final_state
from .initialize_state import create_initial_state
def adjoint_jvp(tape: QuantumTape, tangents: Tuple[Number], state=None):
    """The jacobian vector product used in forward mode calculation of derivatives.

    Implements the adjoint method outlined in
    `Jones and Gacon <https://arxiv.org/abs/2009.02823>`__ to differentiate an input tape.

    After a forward pass, the circuit is reversed by iteratively applying adjoint
    gates to scan backwards through the circuit.

    .. note::

        The adjoint differentiation method has the following restrictions:

        * Cannot differentiate with respect to observables.

        * Observable being measured must have a matrix.

    Args:
        tape (QuantumTape): circuit that the function takes the gradient of
        tangents (Tuple[Number]): gradient vector for input parameters.
        state (TensorLike): the final state of the circuit; if not provided,
            the final state will be computed by executing the tape

    Returns:
        Tuple[Number]: gradient vector for output parameters
    """
    if set(tape.wires) != set(range(tape.num_wires)):
        wire_map = {w: i for i, w in enumerate(tape.wires)}
        tapes, fn = qml.map_wires(tape, wire_map)
        tape = fn(tapes)
    ket = state if state is not None else get_final_state(tape)[0]
    n_obs = len(tape.observables)
    bras = np.empty([n_obs] + [2] * len(tape.wires), dtype=np.complex128)
    for i, obs in enumerate(tape.observables):
        bras[i] = apply_operation(obs, ket)
    param_number = len(tape.get_parameters(trainable_only=False, operations_only=True)) - 1
    trainable_param_number = len(tape.trainable_params) - 1
    tangents_out = np.zeros(n_obs)
    for op in reversed(tape.operations[tape.num_preps:]):
        adj_op = qml.adjoint(op)
        ket = apply_operation(adj_op, ket)
        if op.num_params == 1:
            if param_number in tape.trainable_params:
                if not np.allclose(tangents[trainable_param_number], 0):
                    d_op_matrix = operation_derivative(op)
                    ket_temp = apply_operation(qml.QubitUnitary(d_op_matrix, wires=op.wires), ket)
                    tangents_out += 2 * _dot_product_real(bras, ket_temp, len(tape.wires)) * tangents[trainable_param_number]
                trainable_param_number -= 1
            param_number -= 1
        for i in range(n_obs):
            bras[i] = apply_operation(adj_op, bras[i])
    if n_obs == 1:
        return np.array(tangents_out[0])
    return tuple((np.array(t) for t in tangents_out))