import abc
import collections
from typing import (
import numpy as np
from cirq import devices, ops, protocols, study, value
from cirq.sim.simulation_product_state import SimulationProductState
from cirq.sim.simulation_state import TSimulationState
from cirq.sim.simulation_state_base import SimulationStateBase
from cirq.sim.simulator import (
def _create_simulation_state(self, initial_state: Any, qubits: Sequence['cirq.Qid']) -> SimulationStateBase[TSimulationState]:
    if isinstance(initial_state, SimulationStateBase):
        return initial_state
    classical_data = value.ClassicalDataDictionaryStore()
    if self._split_untangled_states:
        args_map: Dict[Optional['cirq.Qid'], TSimulationState] = {}
        if isinstance(initial_state, int):
            for q in reversed(qubits):
                args_map[q] = self._create_partial_simulation_state(initial_state=initial_state % q.dimension, qubits=[q], classical_data=classical_data)
                initial_state = int(initial_state / q.dimension)
        else:
            args = self._create_partial_simulation_state(initial_state=initial_state, qubits=qubits, classical_data=classical_data)
            for q in qubits:
                args_map[q] = args
        args_map[None] = self._create_partial_simulation_state(0, (), classical_data)
        return SimulationProductState(args_map, qubits, self._split_untangled_states, classical_data=classical_data)
    else:
        return self._create_partial_simulation_state(initial_state=initial_state, qubits=qubits, classical_data=classical_data)