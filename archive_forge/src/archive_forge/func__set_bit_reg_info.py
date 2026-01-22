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
def _set_bit_reg_info(self, wire_map, qubits_dict, clbits_dict, glob_data):
    """Get all the info for drawing bit/reg names and numbers"""
    longest_wire_label_width = 0
    glob_data['n_lines'] = 0
    initial_qbit = ' $|0\\rangle$' if self._initial_state else ''
    initial_cbit = ' 0' if self._initial_state else ''
    idx = 0
    pos = y_off = -len(self._qubits) + 1
    for ii, wire in enumerate(wire_map):
        if isinstance(wire, ClassicalRegister):
            if wire[0] not in self._clbits:
                continue
            register = wire
            index = wire_map[wire]
        else:
            if wire not in self._qubits + self._clbits:
                continue
            register, bit_index, reg_index = get_bit_reg_index(self._circuit, wire)
            index = bit_index if register is None else reg_index
        wire_label = get_wire_label('mpl', register, index, layout=self._layout, cregbundle=self._cregbundle)
        initial_bit = initial_qbit if isinstance(wire, Qubit) else initial_cbit
        if isinstance(wire, Qubit) or register is None or (not self._cregbundle):
            wire_label = '$' + wire_label + '$'
        wire_label += initial_bit
        reg_size = 0 if register is None or isinstance(wire, ClassicalRegister) else register.size
        reg_remove_under = 0 if reg_size < 2 else 1
        text_width = self._get_text_width(wire_label, glob_data, self._style['fs'], reg_remove_under=reg_remove_under) * 1.15
        if text_width > longest_wire_label_width:
            longest_wire_label_width = text_width
        if isinstance(wire, Qubit):
            pos = -ii
            qubits_dict[ii] = {'y': pos, 'wire_label': wire_label}
            glob_data['n_lines'] += 1
        else:
            if not self._cregbundle or register is None or (self._cregbundle and isinstance(wire, ClassicalRegister)):
                glob_data['n_lines'] += 1
                idx += 1
            pos = y_off - idx
            clbits_dict[ii] = {'y': pos, 'wire_label': wire_label, 'register': register}
    glob_data['x_offset'] = -1.2 + longest_wire_label_width