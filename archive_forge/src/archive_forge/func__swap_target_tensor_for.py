from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, TYPE_CHECKING, Union
import numpy as np
from cirq import linalg, protocols, qis, sim
from cirq._compat import proper_repr
from cirq.linalg import transformations
from cirq.sim.simulation_state import SimulationState, strat_act_on_from_apply_decompose
def _swap_target_tensor_for(self, new_target_tensor: np.ndarray):
    """Gives a new state vector for the system.

        Typically, the new state vector should be `args.available_buffer` where
        `args` is this `cirq.StateVectorSimulationState` instance.

        Args:
            new_target_tensor: The new system state. Must have the same shape
                and dtype as the old system state.
        """
    if new_target_tensor is self._buffer:
        self._buffer = self._state_vector
    self._state_vector = new_target_tensor