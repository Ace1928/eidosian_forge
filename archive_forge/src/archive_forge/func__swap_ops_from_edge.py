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
def _swap_ops_from_edge(edge, state):
    """Generate list of ops to implement a SWAP gate along a coupling edge."""
    device_qreg = state.register
    qreg_edge = tuple((device_qreg[i] for i in edge))
    return [DAGOpNode(op=SwapGate(), qargs=qreg_edge, cargs=())]