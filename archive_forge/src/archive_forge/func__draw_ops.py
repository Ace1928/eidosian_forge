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
def _draw_ops(self, nodes, node_data, wire_map, outer_circuit, layer_widths, qubits_dict, clbits_dict, glob_data, verbose=False):
    """Draw the gates in the circuit"""
    self._add_nodes_and_coords(nodes, node_data, wire_map, outer_circuit, layer_widths, qubits_dict, clbits_dict, glob_data)
    prev_x_index = -1
    for layer in nodes:
        l_width = []
        curr_x_index = prev_x_index + 1
        for node in layer:
            op = node.op
            self._get_colors(node, node_data)
            if verbose:
                print(op)
            if getattr(op, 'condition', None) or isinstance(op, SwitchCaseOp):
                cond_xy = [self._plot_coord(node_data[node].x_index, clbits_dict[ii]['y'], layer_widths[node][0], glob_data, isinstance(op, ControlFlowOp)) for ii in clbits_dict]
                self._condition(node, node_data, wire_map, outer_circuit, cond_xy, glob_data)
            mod_control = None
            if getattr(op, 'modifiers', None):
                canonical_modifiers = _canonicalize_modifiers(op.modifiers)
                for modifier in canonical_modifiers:
                    if isinstance(modifier, ControlModifier):
                        mod_control = modifier
                        break
            if isinstance(op, Measure):
                self._measure(node, node_data, outer_circuit, glob_data)
            elif getattr(op, '_directive', False):
                if self._plot_barriers:
                    self._barrier(node, node_data, glob_data)
            elif isinstance(op, ControlFlowOp):
                self._flow_op_gate(node, node_data, glob_data)
            elif len(node_data[node].q_xy) == 1 and (not node.cargs):
                self._gate(node, node_data, glob_data)
            elif isinstance(op, ControlledGate) or mod_control:
                self._control_gate(node, node_data, glob_data, mod_control)
            else:
                self._multiqubit_gate(node, node_data, glob_data)
            if not node_data[node].inside_flow:
                l_width.append(layer_widths[node][0])
        barrier_offset = 0
        if not self._plot_barriers:
            barrier_offset = -1 if all((getattr(nd.op, '_directive', False) for nd in layer)) else 0
        prev_x_index = curr_x_index + (max(l_width) if l_width else 0) + barrier_offset - 1