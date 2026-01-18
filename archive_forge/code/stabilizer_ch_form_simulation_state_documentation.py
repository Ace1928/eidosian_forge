from typing import Optional, Sequence, TYPE_CHECKING, Union
import numpy as np
from cirq._compat import proper_repr
from cirq.sim.clifford import stabilizer_state_ch_form
from cirq.sim.clifford.stabilizer_simulation_state import StabilizerSimulationState
Initializes with the given state and the axes for the operation.

        Args:
            qubits: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            prng: The pseudo random number generator to use for probabilistic
                effects.
            initial_state: The initial state for the simulation. This can be a
                full CH form passed by reference which will be modified inplace,
                or a big-endian int in the computational basis. If the state is
                an integer, qubits must be provided in order to determine
                array sizes.
            classical_data: The shared classical data container for this
                simulation.

        Raises:
            ValueError: If initial state is an integer but qubits are not
                provided.
        