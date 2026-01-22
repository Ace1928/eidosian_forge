from dataclasses import replace
from functools import partial
from numbers import Number
from typing import Union, Callable, Tuple, Optional, Sequence
import concurrent.futures
import inspect
import logging
import numpy as np
import pennylane as qml
from pennylane.ops.op_math.condition import Conditional
from pennylane.measurements.mid_measure import MidMeasureMP
from pennylane.tape import QuantumTape
from pennylane.typing import Result, ResultBatch
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.transforms.core import TransformProgram
from . import Device
from .modifiers import single_tape_support, simulator_tracking
from .preprocess import (
from .execution_config import ExecutionConfig, DefaultExecutionConfig
from .qubit.simulate import simulate, get_final_state, measure_final_state
from .qubit.adjoint_jacobian import adjoint_jacobian, adjoint_vjp, adjoint_jvp
@qml.transform
def adjoint_state_measurements(tape: QuantumTape, device_vjp=False) -> (Tuple[QuantumTape], Callable):
    """Perform adjoint measurement preprocessing.

    * Allows a tape with only expectation values through unmodified
    * Raises an error if non-expectation value measurements exist and any have diagonalizing gates
    * Turns the circuit into a state measurement + classical postprocesssing for arbitrary measurements

    Args:
        tape (QuantumTape): the input circuit

    """
    if all((isinstance(m, qml.measurements.ExpectationMP) for m in tape.measurements)):
        return ((tape,), null_postprocessing)
    if any((len(m.diagonalizing_gates()) > 0 for m in tape.measurements)):
        raise qml.DeviceError('adjoint diff supports either all expectation values or only measurements without observables.')
    params = tape.get_parameters()
    if device_vjp:
        for p in params:
            if qml.math.requires_grad(p) and qml.math.get_interface(p) == 'tensorflow' and (qml.math.get_dtype_name(p) in {'float32', 'complex64'}):
                raise ValueError('tensorflow with adjoint differentiation of the state requires float64 or complex128 parameters.')
    complex_data = [qml.math.cast(p, complex) for p in params]
    tape = tape.bind_new_parameters(complex_data, list(range(len(params))))
    new_mp = qml.measurements.StateMP(wires=tape.wires)
    state_tape = qml.tape.QuantumScript(tape.operations, [new_mp])
    return ((state_tape,), partial(all_state_postprocessing, measurements=tape.measurements, wire_order=tape.wires))