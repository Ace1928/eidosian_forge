import abc
import collections
from typing import (
import numpy as np
from cirq import circuits, ops, protocols, study, value, work
from cirq.sim.simulation_state_base import SimulationStateBase
def _compute_amplitudes_sweep_to_iter(self, program: 'cirq.AbstractCircuit', bitstrings: Sequence[int], params: 'cirq.Sweepable', qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT) -> Iterator[Sequence[complex]]:
    if type(self).compute_amplitudes_sweep == SimulatesAmplitudes.compute_amplitudes_sweep:
        raise RecursionError('Must define either compute_amplitudes_sweep or compute_amplitudes_sweep_iter.')
    yield from self.compute_amplitudes_sweep(program, bitstrings, params, qubit_order)