from numbers import Number
from typing import Tuple
import numpy as np
import pennylane as qml
from pennylane.operation import operation_derivative
from pennylane.tape import QuantumTape
from .apply_operation import apply_operation
from .simulate import get_final_state
from .initialize_state import create_initial_state
The vector jacobian product used in reverse-mode differentiation.

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
        cotangents (Tuple[Number]): gradient vector for output parameters. For computing
            the full Jacobian, the cotangents can be batched to vectorize the computation.
            In this case, the cotangents can have the following shapes. ``batch_size``
            below refers to the number of entries in the Jacobian:

            * For a state measurement, cotangents must have shape ``(batch_size, 2 ** n_wires)``.
            * For ``n`` expectation values, the cotangents must have shape ``(n, batch_size)``.
              If ``n = 1``, then the shape must be ``(batch_size,)``.

        state (TensorLike): the final state of the circuit; if not provided,
            the final state will be computed by executing the tape

    Returns:
        Tuple[Number]: gradient vector for input parameters
    