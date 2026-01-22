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
def _sidetext(self, node, node_data, xy, tc=None, text=''):
    """Draw the sidetext for symmetric gates"""
    xpos, ypos = xy
    xp = xpos + 0.11 + node_data[node].width / 2
    self._ax.text(xp, ypos + HIG, text, ha='center', va='top', fontsize=self._style['sfs'], color=tc, clip_on=True, zorder=PORDER_TEXT)