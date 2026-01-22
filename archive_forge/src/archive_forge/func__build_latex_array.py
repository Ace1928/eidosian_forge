import io
import itertools
import math
import re
from warnings import warn
import numpy as np
from qiskit.circuit import Clbit, Qubit, ClassicalRegister
from qiskit.circuit.classical import expr
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.library.standard_gates import SwapGate, XGate, ZGate, RZZGate, U1Gate, PhaseGate
from qiskit.circuit.measure import Measure
from qiskit.circuit.tools.pi_check import pi_check
from .qcstyle import load_style
from ._utils import (
def _build_latex_array(self):
    """Returns an array of strings containing \\LaTeX for this circuit."""
    column = 1
    if self._cregbundle and (self._nodes and self._nodes[0] and (self._nodes[0][0].op.name == 'measure' or getattr(self._nodes[0][0].op, 'condition', None))):
        column += 1
    for layer in self._nodes:
        num_cols_layer = 1
        for node in layer:
            op = node.op
            num_cols_op = 1
            wire_list = [self._wire_map[qarg] for qarg in node.qargs if qarg in self._qubits]
            if getattr(op, 'condition', None):
                if isinstance(op.condition, expr.Expr):
                    warn('ignoring expression condition, which is not supported yet')
                else:
                    self._add_condition(op, wire_list, column)
            if isinstance(op, Measure):
                self._build_measure(node, column)
            elif getattr(op, '_directive', False):
                self._build_barrier(node, column)
            else:
                gate_text, _, _ = get_gate_ctrl_text(op, 'latex', style=self._style)
                gate_text += get_param_str(op, 'latex', ndigits=4)
                gate_text = generate_latex_label(gate_text)
                if node.cargs:
                    cwire_list = [self._wire_map[carg] for carg in node.cargs if carg in self._clbits]
                else:
                    cwire_list = []
                if len(wire_list) == 1 and (not node.cargs):
                    self._latex[wire_list[0]][column] = '\\gate{%s}' % gate_text
                elif isinstance(op, ControlledGate):
                    num_cols_op = self._build_ctrl_gate(op, gate_text, wire_list, column)
                else:
                    num_cols_op = self._build_multi_gate(op, gate_text, wire_list, cwire_list, column)
            num_cols_layer = max(num_cols_layer, num_cols_op)
        column += num_cols_layer