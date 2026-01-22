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
def _build_multi_gate(self, op, gate_text, wire_list, cwire_list, col):
    """Add a multiple wire gate to the _latex list"""
    cwire_start = len(self._qubits)
    num_cols_op = 1
    if isinstance(op, (SwapGate, RZZGate)):
        num_cols_op = self._build_symmetric_gate(op, gate_text, wire_list, col)
    else:
        wire_min = min(wire_list)
        wire_max = max(wire_list)
        if cwire_list and (not self._cregbundle):
            wire_max = max(cwire_list)
        wire_ind = wire_list.index(wire_min)
        self._latex[wire_min][col] = f'\\multigate{{{wire_max - wire_min}}}{{{gate_text}}}_' + '<' * (len(str(wire_ind)) + 2) + '{%s}' % wire_ind
        for wire in range(wire_min + 1, wire_max + 1):
            if wire < cwire_start:
                ghost_box = '\\ghost{%s}' % gate_text
                if wire in wire_list:
                    wire_ind = wire_list.index(wire)
            else:
                ghost_box = '\\cghost{%s}' % gate_text
                if wire in cwire_list:
                    wire_ind = cwire_list.index(wire)
            if wire in wire_list + cwire_list:
                self._latex[wire][col] = ghost_box + '_' + '<' * (len(str(wire_ind)) + 2) + '{%s}' % wire_ind
            else:
                self._latex[wire][col] = ghost_box
    return num_cols_op