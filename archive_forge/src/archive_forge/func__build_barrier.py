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
def _build_barrier(self, node, col):
    """Build a partial or full barrier if plot_barriers set"""
    if self._plot_barriers:
        indexes = [self._wire_map[qarg] for qarg in node.qargs if qarg in self._qubits]
        indexes.sort()
        first = last = indexes[0]
        for index in indexes[1:]:
            if index - 1 == last:
                last = index
            else:
                pos = self._wire_map[self._qubits[first]]
                self._latex[pos][col - 1] += ' \\barrier[0em]{' + str(last - first) + '}'
                self._latex[pos][col] = '\\qw'
                first = last = index
        pos = self._wire_map[self._qubits[first]]
        self._latex[pos][col - 1] += ' \\barrier[0em]{' + str(last - first) + '}'
        if node.op.label is not None:
            pos = indexes[0]
            label = node.op.label.replace(' ', '\\,')
            self._latex[pos][col] = '\\cds{0}{^{\\mathrm{%s}}}' % label