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
class BoxOnQuWireMid(BoxOnWireMid, BoxOnQuWire):
    """Draws the middle part of a box that affects more than one quantum wire"""

    def __init__(self, label, input_length, order, wire_label='', control_label=None):
        super().__init__(label, input_length, order, wire_label=wire_label)
        if control_label:
            self.mid_format = f'{control_label}{self.wire_label} %s ├'
        else:
            self.mid_format = f'┤{self.wire_label} %s ├'