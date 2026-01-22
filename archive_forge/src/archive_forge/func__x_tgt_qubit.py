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
def _x_tgt_qubit(self, xy, glob_data, ec=None, ac=None):
    """Draw the cnot target symbol"""
    linewidth = self._lwidth2
    xpos, ypos = xy
    box = glob_data['patches_mod'].Circle(xy=(xpos, ypos), radius=HIG * 0.35, fc=ec, ec=ec, linewidth=linewidth, zorder=PORDER_GATE)
    self._ax.add_patch(box)
    self._ax.plot([xpos, xpos], [ypos - 0.2 * HIG, ypos + 0.2 * HIG], color=ac, linewidth=linewidth, zorder=PORDER_GATE_PLUS)
    self._ax.plot([xpos - 0.2 * HIG, xpos + 0.2 * HIG], [ypos, ypos], color=ac, linewidth=linewidth, zorder=PORDER_GATE_PLUS)