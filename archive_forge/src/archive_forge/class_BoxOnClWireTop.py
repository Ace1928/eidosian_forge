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
class BoxOnClWireTop(MultiBox, BoxOnClWire):
    """Draws the top part of a conditional box that affects more than one classical wire"""

    def __init__(self, label='', top_connect=None, wire_label=''):
        super().__init__(label)
        self.wire_label = wire_label
        self.mid_content = ''
        self.bot_format = '│ %s │'
        self.top_connect = top_connect if top_connect else '─'
        self.bot_connect = self.bot_pad = ' '