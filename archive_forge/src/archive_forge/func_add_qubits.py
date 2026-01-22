from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, qis, sim
from cirq._compat import proper_repr
from cirq.linalg import transformations
from cirq.sim.simulation_state import SimulationState, strat_act_on_from_apply_decompose
def add_qubits(self, qubits: Sequence['cirq.Qid']):
    ret = super().add_qubits(qubits)
    return self.kronecker_product(type(self)(qubits=qubits), inplace=True) if ret is NotImplemented else ret