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
def _build_symmetric_gate(self, op, gate_text, wire_list, col):
    """Add symmetric gates for cu1, cp, swap, and rzz"""
    wire_max = max(wire_list)
    wire_next_last = wire_list[-2]
    wire_last = wire_list[-1]
    base_op = None if not hasattr(op, 'base_gate') else op.base_gate
    if isinstance(op, SwapGate) or (base_op and isinstance(base_op, SwapGate)):
        self._latex[wire_next_last][col] = '\\qswap'
        self._latex[wire_last][col] = '\\qswap \\qwx[' + str(wire_next_last - wire_last) + ']'
        return 1
    if isinstance(op, RZZGate) or (base_op and isinstance(base_op, RZZGate)):
        ctrl_bit = '1'
    else:
        ctrl_bit = f'{op.ctrl_state:b}'.rjust(1, '0')[::-1]
    control = '\\ctrlo' if ctrl_bit == '0' else '\\ctrl'
    self._latex[wire_next_last][col] = f'{control}' + ('{' + str(wire_last - wire_next_last) + '}')
    self._latex[wire_last][col] = '\\control \\qw'
    self._latex[wire_max - 1][col + 1] = '\\dstick{\\hspace{2.0em}%s} \\qw' % gate_text
    return 4