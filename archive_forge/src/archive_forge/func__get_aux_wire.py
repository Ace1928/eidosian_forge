from typing import Sequence, Callable
import functools
from functools import partial
import warnings
import numpy as np
import pennylane as qml
from pennylane.circuit_graph import LayerData
from pennylane.queuing import WrappedObj
from pennylane.transforms import transform
def _get_aux_wire(aux_wire, tape, device_wires):
    """Determine an unused wire to be used as auxiliary wire for Hadamard tests.

    Args:
        aux_wire (object): Input auxiliary wire. May be one of a variety of input formats:
            If ``None``, try to infer a reasonable choice based on the number of wires used
            in the ``tape``, and based on ``device_wires``, if they are not ``None``.
            If an ``int``, a ``str`` or a ``Sequence``, convert the input to a ``Wires``
            object and take the first entry of the result. This leads to consistent behaviour
            between ``_get_aux_wire`` and the ``Wires`` class.
            If a ``Wires`` instance already, the conversion to such an instance is performed
            trivially as well (also see the source code of ``~.Wires``).
        tape (pennylane.tape.QuantumTape): Tape to infer the wire for
        device_wires (.wires.Wires): Wires of the device that is going to be used for the
            metric tensor. Facilitates finding a default for ``aux_wire`` if ``aux_wire``
            is ``None`` .

    Returns:
        object: The auxiliary wire to be used. Equals ``aux_wire`` if it was not ``None``\\ ,
        and an often reasonable choice else.
    """
    if aux_wire is not None:
        aux_wire = qml.wires.Wires(aux_wire)[0]
        if aux_wire in tape.wires:
            msg = 'The requested auxiliary wire is already in use by the circuit.'
            raise qml.wires.WireError(msg)
        if device_wires is None or aux_wire in device_wires:
            return aux_wire
        raise qml.wires.WireError('The requested auxiliary wire does not exist on the used device.')
    if device_wires is not None:
        if len(device_wires) == len(tape.wires):
            raise qml.wires.WireError('The device has no free wire for the auxiliary wire.')
        unused_wires = qml.wires.Wires(device_wires.toset().difference(tape.wires.toset()))
        return unused_wires[0]
    _wires = tape.wires
    for _aux in range(tape.num_wires):
        if _aux not in _wires:
            return _aux
    return tape.num_wires