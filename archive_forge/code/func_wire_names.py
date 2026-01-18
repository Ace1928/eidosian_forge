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
def wire_names(self, with_initial_state=False):
    """Returns a list of names for each wire.

        Args:
            with_initial_state (bool): Optional (Default: False). If true, adds
                the initial value to the name.

        Returns:
            List: The list of wire names.
        """
    if with_initial_state:
        initial_qubit_value = '|0>'
        initial_clbit_value = '0 '
    else:
        initial_qubit_value = ''
        initial_clbit_value = ''
    self._wire_map = get_wire_map(self._circuit, self.qubits + self.clbits, self.cregbundle)
    wire_labels = []
    for wire in self._wire_map:
        if isinstance(wire, ClassicalRegister):
            register = wire
            index = self._wire_map[wire]
        else:
            register, bit_index, reg_index = get_bit_reg_index(self._circuit, wire)
            index = bit_index if register is None else reg_index
        wire_label = get_wire_label('text', register, index, layout=self.layout, cregbundle=self.cregbundle)
        wire_label += ' ' if self.layout is not None and isinstance(wire, Qubit) else ': '
        cregb_add = ''
        if isinstance(wire, Qubit):
            initial_bit_value = initial_qubit_value
        else:
            initial_bit_value = initial_clbit_value
            if self.cregbundle and register is not None:
                cregb_add = str(register.size) + '/'
        wire_labels.append(wire_label + initial_bit_value + cregb_add)
    return wire_labels