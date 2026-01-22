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
def _ctrl_qubit(self, xy, glob_data, fc=None, ec=None, tc=None, text='', text_top=None):
    """Draw a control circle and if top or bottom control, draw control label"""
    xpos, ypos = xy
    box = glob_data['patches_mod'].Circle(xy=(xpos, ypos), radius=WID * 0.15, fc=fc, ec=ec, linewidth=self._lwidth15, zorder=PORDER_GATE)
    self._ax.add_patch(box)
    label_padding = 0.7
    if text is not None:
        text_lines = text.count('\n')
        if not text.endswith('(cal)\n'):
            for _ in range(text_lines):
                label_padding += 0.3
    if text_top is None:
        return
    ctrl_ypos = ypos + label_padding * HIG if text_top else ypos - 0.3 * HIG
    self._ax.text(xpos, ctrl_ypos, text, ha='center', va='top', fontsize=self._style['sfs'], color=tc, clip_on=True, zorder=PORDER_TEXT)