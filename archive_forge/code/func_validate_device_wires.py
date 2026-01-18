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
def validate_device_wires(tape: qml.tape.QuantumTape, wires: Optional[qml.wires.Wires]=None, name: str='device') -> (Sequence[qml.tape.QuantumTape], Callable):
    """Validates that all wires present in the tape are in the set of provided wires. Adds the
    device wires to measurement processes like :class:`~.measurements.StateMP` that are broadcasted
    across all available wires.

    Args:
        tape (QuantumTape or QNode or Callable): a quantum circuit.
        wires=None (Optional[Wires]): the allowed wires. Wires of ``None`` allows any wires
            to be present in the tape.
        name="device" (str): the name of the device to use in error messages.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        The unaltered input circuit. The output type is explained in :func:`qml.transform <pennylane.transform>`.

    Raises:
        WireError: if the tape has a wire not present in the provided wires.
    """
    if wires:
        if (extra_wires := (set(tape.wires) - set(wires))):
            raise WireError(f'Cannot run circuit(s) on {name} as they contain wires not found on the device: {extra_wires}')
        measurements = tape.measurements.copy()
        modified = False
        for m_idx, mp in enumerate(measurements):
            if not mp.obs and (not mp.wires):
                modified = True
                new_mp = copy(mp)
                new_mp._wires = wires
                measurements[m_idx] = new_mp
        if modified:
            tape = type(tape)(tape.operations, measurements, shots=tape.shots)
    return ((tape,), null_postprocessing)