from dataclasses import dataclass
from typing import Callable, List, Union
import numpy as np
import pennylane as qml
from pennylane.wires import Wires
from pennylane.operation import Operator
from pennylane.ops.qubit.hamiltonian import Hamiltonian
from .parametrized_hamiltonian import ParametrizedHamiltonian
class AmplitudeAndPhase:
    """Class storing combined amplitude and phase callable if either or both
    of amplitude or phase are callable."""

    def __init__(self, trig_fn, amp, phase, hz_to_rads=2 * np.pi):
        self.amp_is_callable = callable(amp)
        self.phase_is_callable = callable(phase)

        def callable_amp_and_phase(params, t):
            return hz_to_rads * amp(params[0], t) * trig_fn(phase(params[1], t))

        def callable_amp(params, t):
            return hz_to_rads * amp(params, t) * trig_fn(phase)

        def callable_phase(params, t):
            return hz_to_rads * amp * trig_fn(phase(params, t))
        if self.amp_is_callable and self.phase_is_callable:
            self.func = callable_amp_and_phase
        elif self.amp_is_callable:
            self.func = callable_amp
        elif self.phase_is_callable:
            self.func = callable_phase

    def __call__(self, params, t):
        return self.func(params, t)