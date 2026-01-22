import datetime
import functools
from typing import Dict, List, NamedTuple, Optional, Tuple, TYPE_CHECKING
from cirq.protocols.json_serialization import ObjectFactory
def _boolean_hamiltonian_gate_op(qubit_map, boolean_strs, theta):
    return cirq.BooleanHamiltonianGate(parameter_names=list(qubit_map.keys()), boolean_strs=boolean_strs, theta=theta).on(*qubit_map.values())