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
def _transform_gate_for_system(gate, state):
    """Return op implementing a virtual gate on given layout."""
    mapped_op_node = copy.copy(_first_op_node(gate['graph']))
    device_qreg = state.register
    layout_map = state.layout._v2p
    mapped_op_node.qargs = tuple((device_qreg[layout_map[a]] for a in mapped_op_node.qargs))
    return mapped_op_node