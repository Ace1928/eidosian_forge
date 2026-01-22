import collections
import itertools
import re
from io import StringIO
import numpy as np
from qiskit.circuit import (
from qiskit.circuit.controlflow import condition_resources
from qiskit.circuit.classical import expr
from qiskit.circuit.annotated_operation import _canonicalize_modifiers, ControlModifier
from qiskit.circuit.library import Initialize
from qiskit.circuit.library.standard_gates import (
from qiskit.qasm3.exporter import QASM3Builder
from qiskit.qasm3.printer import BasicPrinter
from qiskit.circuit.tools.pi_check import pi_check
from qiskit.utils import optionals as _optionals
from .qcstyle import load_style
from ._utils import (
from ..utils import matplotlib_close_if_inline
def _add_nodes_and_coords(self, nodes, node_data, wire_map, outer_circuit, layer_widths, qubits_dict, clbits_dict, glob_data):
    """Add the nodes from ControlFlowOps and their coordinates to the main circuit"""
    for flow_drawers in self._flow_drawers.values():
        for flow_drawer in flow_drawers:
            nodes += flow_drawer._nodes
            flow_drawer._get_coords(node_data, flow_drawer._flow_wire_map, outer_circuit, layer_widths, qubits_dict, clbits_dict, glob_data, flow_parent=flow_drawer._flow_parent)
            flow_drawer._add_nodes_and_coords(nodes, node_data, wire_map, outer_circuit, layer_widths, qubits_dict, clbits_dict, glob_data)