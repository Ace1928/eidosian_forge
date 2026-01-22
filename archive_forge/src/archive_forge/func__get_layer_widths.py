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
def _get_layer_widths(self, node_data, wire_map, outer_circuit, glob_data, builder=None):
    """Compute the layer_widths for the layers"""
    layer_widths = {}
    for layer_num, layer in enumerate(self._nodes):
        widest_box = WID
        for i, node in enumerate(layer):
            if i != 0:
                layer_num = -1
            layer_widths[node] = [1, layer_num, self._flow_parent]
            op = node.op
            node_data[node] = NodeData()
            node_data[node].width = WID
            num_ctrl_qubits = getattr(op, 'num_ctrl_qubits', 0)
            if getattr(op, '_directive', False) and (not op.label or not self._plot_barriers) or isinstance(op, Measure):
                node_data[node].raw_gate_text = op.name
                continue
            base_type = getattr(op, 'base_gate', None)
            gate_text, ctrl_text, raw_gate_text = get_gate_ctrl_text(op, 'mpl', style=self._style, calibrations=self._calibrations)
            node_data[node].gate_text = gate_text
            node_data[node].ctrl_text = ctrl_text
            node_data[node].raw_gate_text = raw_gate_text
            node_data[node].param_text = ''
            if len(node.qargs) - num_ctrl_qubits == 1 and len(gate_text) < 3 and (len(getattr(op, 'params', [])) == 0) and (ctrl_text is None):
                continue
            if isinstance(op, SwapGate) or isinstance(base_type, SwapGate):
                continue
            ctrl_width = self._get_text_width(ctrl_text, glob_data, fontsize=self._style['sfs']) - 0.05
            if len(getattr(op, 'params', [])) > 0 and (not any((isinstance(param, np.ndarray) for param in op.params))) and (not any((isinstance(param, QuantumCircuit) for param in op.params))):
                param_text = get_param_str(op, 'mpl', ndigits=3)
                if isinstance(op, Initialize):
                    param_text = f'$[{param_text.replace('$', '')}]$'
                node_data[node].param_text = param_text
                raw_param_width = self._get_text_width(param_text, glob_data, fontsize=self._style['sfs'], param=True)
                param_width = raw_param_width + 0.08
            else:
                param_width = raw_param_width = 0.0
            if isinstance(op, RZZGate) or isinstance(base_type, (U1Gate, PhaseGate, RZZGate)):
                if isinstance(base_type, PhaseGate):
                    gate_text = 'P'
                raw_gate_width = self._get_text_width(gate_text + ' ()', glob_data, fontsize=self._style['sfs']) + raw_param_width
                gate_width = (raw_gate_width + 0.08) * 1.58
            elif isinstance(node.op, ControlFlowOp):
                self._flow_drawers[node] = []
                node_data[node].width = []
                node_data[node].nest_depth = 0
                gate_width = 0.0
                expr_width = 0.0
                if isinstance(op, SwitchCaseOp) and isinstance(op.target, expr.Expr) or (getattr(op, 'condition', None) and isinstance(op.condition, expr.Expr)):
                    condition = op.target if isinstance(op, SwitchCaseOp) else op.condition
                    if builder is None:
                        builder = QASM3Builder(outer_circuit, includeslist=('stdgates.inc',), basis_gates=('U',), disable_constants=False, allow_aliasing=False)
                        builder.build_classical_declarations()
                    stream = StringIO()
                    BasicPrinter(stream, indent='  ').visit(builder.build_expression(condition))
                    expr_text = stream.getvalue()
                    if len(expr_text) > self._expr_len:
                        expr_text = expr_text[:self._expr_len] + '...'
                    node_data[node].expr_text = expr_text
                    expr_width = self._get_text_width(node_data[node].expr_text, glob_data, fontsize=self._style['sfs'])
                    node_data[node].expr_width = int(expr_width)
                circuit_list = list(node.op.blocks)
                if isinstance(op, ForLoopOp):
                    node_data[node].indexset = op.params[0]
                elif isinstance(op, SwitchCaseOp):
                    node_data[node].jump_values = []
                    cases = list(op.cases_specifier())
                    circuit_list.insert(0, cases[0][1].copy_empty_like())
                    for jump_values, _ in cases:
                        node_data[node].jump_values.append(jump_values)
                for circ_num, circuit in enumerate(circuit_list):
                    raw_gate_width = expr_width if circ_num == 0 else 0.0
                    if self._flow_parent is not None:
                        node_data[node].nest_depth = node_data[self._flow_parent].nest_depth + 1
                    flow_wire_map = wire_map.copy()
                    flow_wire_map.update({inner: wire_map[outer] for outer, inner in zip(node.qargs, circuit.qubits)})
                    for outer, inner in zip(node.cargs, circuit.clbits):
                        if self._cregbundle and (in_reg := get_bit_register(outer_circuit, inner)) is not None:
                            out_reg = get_bit_register(outer_circuit, outer)
                            flow_wire_map.update({in_reg: wire_map[out_reg]})
                        else:
                            flow_wire_map.update({inner: wire_map[outer]})
                    qubits, clbits, flow_nodes = _get_layered_instructions(circuit, wire_map=flow_wire_map)
                    flow_drawer = MatplotlibDrawer(qubits, clbits, flow_nodes, circuit, style=self._style, plot_barriers=self._plot_barriers, fold=self._fold, cregbundle=self._cregbundle)
                    flow_drawer._flow_parent = node
                    flow_drawer._flow_wire_map = flow_wire_map
                    self._flow_drawers[node].append(flow_drawer)
                    flow_widths = flow_drawer._get_layer_widths(node_data, flow_wire_map, outer_circuit, glob_data, builder)
                    layer_widths.update(flow_widths)
                    for flow_layer in flow_nodes:
                        for flow_node in flow_layer:
                            node_data[flow_node].circ_num = circ_num
                    for width, layer_num, flow_parent in flow_widths.values():
                        if layer_num != -1 and flow_parent == flow_drawer._flow_parent:
                            raw_gate_width += width
                    gate_width += raw_gate_width + (1.0 if circ_num > 0 else 0.0)
                    if circ_num > 0:
                        raw_gate_width += 0.045
                    if not isinstance(op, ForLoopOp) and circ_num == 0:
                        node_data[node].width.append(raw_gate_width - expr_width % 1)
                    else:
                        node_data[node].width.append(raw_gate_width)
            else:
                raw_gate_width = self._get_text_width(gate_text, glob_data, fontsize=self._style['fs'])
                gate_width = raw_gate_width + 0.1
                if len(node.qargs) - num_ctrl_qubits > 1:
                    gate_width += 0.21
            box_width = max(gate_width, ctrl_width, param_width, WID)
            if box_width > widest_box:
                widest_box = box_width
            if not isinstance(node.op, ControlFlowOp):
                node_data[node].width = max(raw_gate_width, raw_param_width)
        for node in layer:
            layer_widths[node][0] = int(widest_box) + 1
    return layer_widths