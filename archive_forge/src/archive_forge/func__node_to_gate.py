from io import StringIO
from warnings import warn
from shutil import get_terminal_size
import collections
import sys
from qiskit.circuit import Qubit, Clbit, ClassicalRegister
from qiskit.circuit import ControlledGate, Reset, Measure
from qiskit.circuit import ControlFlowOp, WhileLoopOp, IfElseOp, ForLoopOp, SwitchCaseOp
from qiskit.circuit.classical import expr
from qiskit.circuit.controlflow import node_resources
from qiskit.circuit.library.standard_gates import IGate, RZZGate, SwapGate, SXGate, SXdgGate
from qiskit.circuit.annotated_operation import _canonicalize_modifiers, ControlModifier
from qiskit.circuit.tools.pi_check import pi_check
from qiskit.qasm3.exporter import QASM3Builder
from qiskit.qasm3.printer import BasicPrinter
from ._utils import (
from ..exceptions import VisualizationError
def _node_to_gate(self, node, layer, gate_wire_map):
    """Convert a dag op node into its corresponding Gate object, and establish
        any connections it introduces between qubits. gate_wire_map is the flow_wire_map
        if gate is inside a ControlFlowOp, else it's self._wire_map"""
    op = node.op
    current_cons = []
    current_cons_cond = []
    connection_label = None
    conditional = False
    base_gate = getattr(op, 'base_gate', None)
    params = get_param_str(op, 'text', ndigits=5)
    if not isinstance(op, (Measure, SwapGate, Reset)) and (not getattr(op, '_directive', False)):
        gate_text, ctrl_text, _ = get_gate_ctrl_text(op, 'text')
        gate_text = TextDrawing.special_label(op) or gate_text
        gate_text = gate_text + params
    if getattr(op, 'condition', None) is not None:
        current_cons_cond += layer.set_cl_multibox(op.condition, gate_wire_map, top_connect='╨')
        conditional = True

    def add_connected_gate(node, gates, layer, current_cons, gate_wire_map):
        for i, gate in enumerate(gates):
            actual_index = gate_wire_map[node.qargs[i]]
            if actual_index not in [i for i, j in current_cons]:
                layer.set_qubit(node.qargs[i], gate)
                current_cons.append((actual_index, gate))
    mod_control = None
    if getattr(op, 'modifiers', None):
        canonical_modifiers = _canonicalize_modifiers(op.modifiers)
        for modifier in canonical_modifiers:
            if isinstance(modifier, ControlModifier):
                mod_control = modifier
                break
    if isinstance(op, Measure):
        gate = MeasureFrom()
        layer.set_qubit(node.qargs[0], gate)
        register, _, reg_index = get_bit_reg_index(self._circuit, node.cargs[0])
        if self.cregbundle and register is not None:
            layer.set_clbit(node.cargs[0], MeasureTo(str(reg_index)))
        else:
            layer.set_clbit(node.cargs[0], MeasureTo())
    elif getattr(op, '_directive', False):
        if not self.plotbarriers:
            return (layer, current_cons, current_cons_cond, connection_label)
        for i, qubit in enumerate(node.qargs):
            if qubit in self.qubits:
                label = op.label if i == 0 else ''
                layer.set_qubit(qubit, Barrier(label))
    elif isinstance(op, SwapGate):
        gates = [Ex(conditional=conditional) for _ in range(len(node.qargs))]
        add_connected_gate(node, gates, layer, current_cons, gate_wire_map)
    elif isinstance(op, Reset):
        layer.set_qubit(node.qargs[0], ResetDisplay(conditional=conditional))
    elif isinstance(op, RZZGate):
        connection_label = 'ZZ%s' % params
        gates = [Bullet(conditional=conditional), Bullet(conditional=conditional)]
        add_connected_gate(node, gates, layer, current_cons, gate_wire_map)
    elif len(node.qargs) == 1 and (not node.cargs):
        layer.set_qubit(node.qargs[0], BoxOnQuWire(gate_text, conditional=conditional))
    elif isinstance(op, ControlledGate) or mod_control:
        controls_array = TextDrawing.controlled_wires(node, gate_wire_map, ctrl_text, conditional, mod_control)
        gates, controlled_top, controlled_bot, controlled_edge, rest = controls_array
        if mod_control:
            if len(rest) == 1:
                gates.append(BoxOnQuWire(gate_text, conditional=conditional))
            else:
                top_connect = '┴' if controlled_top else None
                bot_connect = '┬' if controlled_bot else None
                indexes = layer.set_qu_multibox(rest, gate_text, conditional=conditional, controlled_edge=controlled_edge, top_connect=top_connect, bot_connect=bot_connect)
                for index in range(min(indexes), max(indexes) + 1):
                    current_cons.append((index, DrawElement('')))
        elif base_gate.name == 'z':
            gates.append(Bullet(conditional=conditional))
        elif base_gate.name in ['u1', 'p']:
            connection_label = f'{base_gate.name.upper()}{params}'
            gates.append(Bullet(conditional=conditional))
        elif base_gate.name == 'swap':
            gates += [Ex(conditional=conditional), Ex(conditional=conditional)]
            add_connected_gate(node, gates, layer, current_cons, gate_wire_map)
        elif base_gate.name == 'rzz':
            connection_label = 'ZZ%s' % params
            gates += [Bullet(conditional=conditional), Bullet(conditional=conditional)]
        elif len(rest) > 1:
            top_connect = '┴' if controlled_top else None
            bot_connect = '┬' if controlled_bot else None
            indexes = layer.set_qu_multibox(rest, gate_text, conditional=conditional, controlled_edge=controlled_edge, top_connect=top_connect, bot_connect=bot_connect)
            for index in range(min(indexes), max(indexes) + 1):
                current_cons.append((index, DrawElement('')))
        else:
            gates.append(BoxOnQuWire(gate_text, conditional=conditional))
        add_connected_gate(node, gates, layer, current_cons, gate_wire_map)
    elif len(node.qargs) >= 2 and (not node.cargs):
        layer.set_qu_multibox(node.qargs, gate_text, conditional=conditional)
    elif node.qargs and node.cargs:
        layer._set_multibox(gate_text, qargs=node.qargs, cargs=node.cargs, conditional=conditional)
    else:
        raise VisualizationError('Text visualizer does not know how to handle this node: ', op.name)
    current_cons.sort(key=lambda tup: tup[0])
    current_cons = [g for q, g in current_cons]
    current_cons_cond.sort(key=lambda tup: tup[0])
    current_cons_cond = [g for c, g in current_cons_cond]
    return (layer, current_cons, current_cons_cond, connection_label)