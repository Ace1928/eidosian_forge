import abc
import collections
from typing import (
import numpy as np
from cirq import devices, ops, protocols, study, value
from cirq.sim.simulation_product_state import SimulationProductState
from cirq.sim.simulation_state import TSimulationState
from cirq.sim.simulation_state_base import SimulationStateBase
from cirq.sim.simulator import (
def _core_iterator(self, circuit: 'cirq.AbstractCircuit', sim_state: SimulationStateBase[TSimulationState], all_measurements_are_terminal: bool=False) -> Iterator[TStepResultBase]:
    """Standard iterator over StepResult from Moments of a Circuit.

        Args:
            circuit: The circuit to simulate.
            sim_state: The initial args for the simulation. The form of
                this state depends on the simulation implementation. See
                documentation of the implementing class for details.
            all_measurements_are_terminal: Whether all measurements in the
                given circuit are terminal.

        Yields:
            StepResults from simulating a Moment of the Circuit.

        Raises:
            TypeError: The simulator encounters an op it does not support.
        """
    if len(circuit) == 0:
        yield self._create_step_result(sim_state)
        return
    noisy_moments = self.noise.noisy_moments(circuit, sorted(circuit.all_qubits()))
    measured: Dict[Tuple['cirq.Qid', ...], bool] = collections.defaultdict(bool)
    for moment in noisy_moments:
        for op in ops.flatten_to_ops(moment):
            try:
                if all_measurements_are_terminal and measured[op.qubits]:
                    continue
                if isinstance(op.gate, ops.MeasurementGate):
                    measured[op.qubits] = True
                    if all_measurements_are_terminal:
                        continue
                protocols.act_on(op, sim_state)
            except TypeError:
                raise TypeError(f"{self.__class__.__name__} doesn't support {op!r}")
        yield self._create_step_result(sim_state)