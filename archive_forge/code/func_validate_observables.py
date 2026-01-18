import os
from typing import Generator, Callable, Union, Sequence, Optional
from copy import copy
import warnings
import pennylane as qml
from pennylane import Snapshot
from pennylane.operation import Tensor, StatePrepBase
from pennylane.measurements import (
from pennylane.typing import ResultBatch, Result
from pennylane import DeviceError
from pennylane import transform
from pennylane.wires import WireError
@transform
def validate_observables(tape: qml.tape.QuantumTape, stopping_condition: Callable[[qml.operation.Operator], bool], name: str='device') -> (Sequence[qml.tape.QuantumTape], Callable):
    """Validates the observables and measurements for a circuit.

    Args:
        tape (QuantumTape or QNode or Callable): a quantum circuit.
        stopping_condition (callable): a function that specifies whether or not an observable is accepted.
        name (str): the name of the device to use in error messages.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[.QuantumTape], function]:

        The unaltered input circuit. The output type is explained in :func:`qml.transform <pennylane.transform>`.

    Raises:
        DeviceError: if an observable is not supported

    **Example:**

    >>> def accepted_observable(obj):
    ...    return obj.name in {"PauliX", "PauliY", "PauliZ"}
    >>> tape = qml.tape.QuantumScript([], [qml.expval(qml.Z(0) + qml.Y(0))])
    >>> validate_observables(tape, accepted_observable)
    DeviceError: Observable <Hamiltonian: terms=2, wires=[0]> not supported on device

    Note that if the observable is a :class:`~.Tensor`, the validation is run on each object in the
    ``Tensor`` instead.

    """
    for m in tape.measurements:
        if m.obs is not None:
            if isinstance(m.obs, Tensor):
                if any((not stopping_condition(o) for o in m.obs.obs)):
                    raise DeviceError(f'Observable {repr(m.obs)} not supported on {name}')
            elif not stopping_condition(m.obs):
                raise DeviceError(f'Observable {repr(m.obs)} not supported on {name}')
    return ((tape,), null_postprocessing)