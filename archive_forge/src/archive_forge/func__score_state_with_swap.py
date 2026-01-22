import collections
import copy
import logging
import math
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler.target import Target
from qiskit.transpiler.passes.layout import disjoint_utils
def _score_state_with_swap(swap, state, gates):
    """Calculate the relative score for a given SWAP.

    Returns:
        float: the score of the given swap.
        Tuple[int, int]: the input swap that should be performed.
        _SystemState: an updated system state with the new layout contained.
    """
    trial_layout = state.layout.copy()
    trial_layout.swap(*swap)
    new_state = state._replace(layout=trial_layout)
    return (_calc_layout_distance(gates, new_state), swap, new_state)