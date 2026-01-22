from typing import Optional, Sequence, TYPE_CHECKING
import numpy as np
from cirq.qis import clifford_tableau
from cirq.sim.clifford.stabilizer_simulation_state import StabilizerSimulationState
Inits CliffordTableauSimulationState.

        Args:
            tableau: The CliffordTableau to act on. Operations are expected to
                perform inplace edits of this object.
            qubits: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            prng: The pseudo random number generator to use for probabilistic
                effects.
            classical_data: The shared classical data container for this
                simulation.
        