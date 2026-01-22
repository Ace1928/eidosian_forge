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
def _control_gate(self, node, node_data, glob_data, mod_control):
    """Draw a controlled gate"""
    op = node.op
    xy = node_data[node].q_xy
    base_type = getattr(op, 'base_gate', None)
    qubit_b = min(xy, key=lambda xy: xy[1])
    qubit_t = max(xy, key=lambda xy: xy[1])
    num_ctrl_qubits = mod_control.num_ctrl_qubits if mod_control else op.num_ctrl_qubits
    num_qargs = len(xy) - num_ctrl_qubits
    ctrl_state = mod_control.ctrl_state if mod_control else op.ctrl_state
    self._set_ctrl_bits(ctrl_state, num_ctrl_qubits, xy, glob_data, ec=node_data[node].ec, tc=node_data[node].tc, text=node_data[node].ctrl_text, qargs=node.qargs)
    self._line(qubit_b, qubit_t, lc=node_data[node].lc)
    if isinstance(op, RZZGate) or isinstance(base_type, (U1Gate, PhaseGate, ZGate, RZZGate)):
        self._symmetric_gate(node, node_data, base_type, glob_data)
    elif num_qargs == 1 and isinstance(base_type, XGate):
        tgt_color = self._style['dispcol']['target']
        tgt = tgt_color if isinstance(tgt_color, str) else tgt_color[0]
        self._x_tgt_qubit(xy[num_ctrl_qubits], glob_data, ec=node_data[node].ec, ac=tgt)
    elif num_qargs == 1:
        self._gate(node, node_data, glob_data, xy[num_ctrl_qubits:][0])
    elif isinstance(base_type, SwapGate):
        self._swap(xy[num_ctrl_qubits:], node, node_data, node_data[node].lc)
    else:
        self._multiqubit_gate(node, node_data, glob_data, xy[num_ctrl_qubits:])