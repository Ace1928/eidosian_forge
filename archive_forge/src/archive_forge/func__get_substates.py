import abc
import collections
from typing import (
import numpy as np
from cirq import devices, ops, protocols, study, value
from cirq.sim.simulation_product_state import SimulationProductState
from cirq.sim.simulation_state import TSimulationState
from cirq.sim.simulation_state_base import SimulationStateBase
from cirq.sim.simulator import (
def _get_substates(self) -> Sequence[TSimulationState]:
    state = self._final_simulator_state
    if isinstance(state, SimulationProductState):
        substates: Dict[TSimulationState, int] = {}
        for q in state.qubits:
            substates[self.get_state_containing_qubit(q)] = 0
        substates[state[None]] = 0
        return tuple(substates.keys())
    return [state.create_merged_state()]