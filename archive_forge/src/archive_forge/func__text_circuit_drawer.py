from __future__ import annotations
import logging
import os
import subprocess
import tempfile
import shutil
import typing
from warnings import warn
from qiskit import user_config
from qiskit.utils import optionals as _optionals
from qiskit.circuit import ControlFlowOp, Measure
from . import latex as _latex
from . import text as _text
from . import matplotlib as _matplotlib
from . import _utils
from ..utils import _trim as trim_image
from ..exceptions import VisualizationError
def _text_circuit_drawer(circuit, filename=None, reverse_bits=False, plot_barriers=True, justify=None, vertical_compression='high', idle_wires=True, with_layout=True, fold=None, initial_state=True, cregbundle=None, encoding=None, wire_order=None, expr_len=30):
    """Draws a circuit using ascii art.

    Args:
        circuit (QuantumCircuit): Input circuit
        filename (str): Optional filename to write the result
        reverse_bits (bool): Rearrange the bits in reverse order.
        plot_barriers (bool): Draws the barriers when they are there.
        justify (str) : `left`, `right` or `none`. Defaults to `left`. Says how
            the circuit should be justified.
        vertical_compression (string): `high`, `medium`, or `low`. It merges the
            lines so the drawing will take less vertical room. Default is `high`.
        idle_wires (bool): Include idle wires. Default is True.
        with_layout (bool): Include layout information with labels on the physical
            layout. Default: True
        fold (int): Optional. Breaks the circuit drawing to this length. This
            is useful when the drawing does not fit in the console. If
            None (default), it will try to guess the console width using
            `shutil.get_terminal_size()`. If you don't want pagination
            at all, set `fold=-1`.
        initial_state (bool): Optional. Adds |0> in the beginning of the line.
            Default: `False`.
        cregbundle (bool): Optional. If set True, bundle classical registers.
            Default: ``True``.
        encoding (str): Optional. Sets the encoding preference of the output.
            Default: ``sys.stdout.encoding``.
        wire_order (list): Optional. A list of integers used to reorder the display
            of the bits. The list must have an entry for every bit with the bits
            in the range 0 to (num_qubits + num_clbits).
        expr_len (int): Optional. The number of characters to display if an :class:`~.expr.Expr`
            is used for the condition in a :class:`.ControlFlowOp`. If this number is exceeded,
            the string will be truncated at that number and '...' added to the end.

    Returns:
        TextDrawing: An instance that, when printed, draws the circuit in ascii art.

    Raises:
        VisualizationError: When the filename extension is not .txt.
    """
    qubits, clbits, nodes = _utils._get_layered_instructions(circuit, reverse_bits=reverse_bits, justify=justify, idle_wires=idle_wires, wire_order=wire_order)
    text_drawing = _text.TextDrawing(qubits, clbits, nodes, circuit, reverse_bits=reverse_bits, initial_state=initial_state, cregbundle=cregbundle, encoding=encoding, with_layout=with_layout, expr_len=expr_len)
    text_drawing.plotbarriers = plot_barriers
    text_drawing.line_length = fold
    text_drawing.vertical_compression = vertical_compression
    if filename:
        text_drawing.dump(filename, encoding=encoding)
    return text_drawing