import abc
import copy
from typing import (
from typing_extensions import Self
import numpy as np
from cirq import ops, protocols, value
from cirq.sim.simulation_state_base import SimulationStateBase
Renames `q1` to `q2`.

        Args:
            q1: The qubit to rename.
            q2: The new name.
            inplace: True to rename the qubit in the current object, False to
                create a copy with the qubit renamed.

        Returns:
            The original object with the qubits renamed if inplace is
            requested, or a copy of the original object with the qubits renamed
            otherwise.

        Raises:
            ValueError: If the qubits are of different dimensionality.
        