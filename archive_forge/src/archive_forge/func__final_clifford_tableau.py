from typing import Any, cast, Optional, Type
import numpy as np
from cirq.circuits.circuit import Circuit
from cirq.devices import LineQubit
from cirq.ops import common_gates
from cirq.ops.dense_pauli_string import DensePauliString
from cirq import protocols
from cirq.qis import clifford_tableau
from cirq.sim import state_vector_simulation_state, final_state_vector
from cirq.sim.clifford import (
def _final_clifford_tableau(circuit: Circuit, qubit_map) -> Optional[clifford_tableau.CliffordTableau]:
    """Evolves a default CliffordTableau through the input circuit.

    Initializes a CliffordTableau with default args for the given qubits and
    evolves it by having each operation act on the tableau.

    Args:
        circuit: An input circuit that acts on the zero state
        qubit_map: A map from qid to the qubit index for the above circuit

    Returns:
        None if any of the operations can not act on a CliffordTableau, returns
        the tableau otherwise."""
    tableau = clifford_tableau.CliffordTableau(len(qubit_map))
    args = clifford_tableau_simulation_state.CliffordTableauSimulationState(tableau=tableau, qubits=list(qubit_map.keys()), prng=np.random.RandomState())
    for op in circuit.all_operations():
        try:
            protocols.act_on(op, args, allow_decompose=True)
        except TypeError:
            return None
    return tableau