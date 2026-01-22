from dataclasses import dataclass
from typing import Callable, List, Union
import numpy as np
import pennylane as qml
from pennylane.wires import Wires
from pennylane.operation import Operator
from pennylane.ops.qubit.hamiltonian import Hamiltonian
from .parametrized_hamiltonian import ParametrizedHamiltonian
def amplitude_and_phase(trig_fn, amp, phase, hz_to_rads=2 * np.pi):
    """Wrapper function for combining amplitude and phase into a single callable
    (or constant if neither amplitude nor phase are callable). The factor of :math:`2 \\pi` converts
    amplitude in Hz to amplitude in radians/second."""
    if not callable(amp) and (not callable(phase)):
        return hz_to_rads * amp * trig_fn(phase)
    return AmplitudeAndPhase(trig_fn, amp, phase, hz_to_rads=hz_to_rads)