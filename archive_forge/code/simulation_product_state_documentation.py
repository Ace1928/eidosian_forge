from collections import abc
from typing import Any, Dict, Generic, Iterator, List, Mapping, Optional, Sequence, TYPE_CHECKING
import numpy as np
from cirq import ops, protocols, value
from cirq.sim.simulation_state import TSimulationState
from cirq.sim.simulation_state_base import SimulationStateBase
Initializes the class.

        Args:
            sim_states: The `SimulationState` dictionary. This will not be
                copied; the original reference will be kept here.
            qubits: The canonical ordering of qubits.
            split_untangled_states: If True, optimizes operations by running
                unentangled qubit sets independently and merging those states
                at the end.
            classical_data: The shared classical data container for this
                simulation.
        