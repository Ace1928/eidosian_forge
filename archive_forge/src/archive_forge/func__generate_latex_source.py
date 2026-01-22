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
def _generate_latex_source(circuit, filename=None, scale=0.7, style=None, reverse_bits=False, plot_barriers=True, justify=None, idle_wires=True, with_layout=True, initial_state=False, cregbundle=None, wire_order=None):
    """Convert QuantumCircuit to LaTeX string.

    Args:
        circuit (QuantumCircuit): a quantum circuit
        scale (float): scaling factor
        style (dict or str): dictionary of style or file name of style file
        filename (str): optional filename to write latex
        reverse_bits (bool): When set to True reverse the bit order inside
            registers for the output visualization.
        plot_barriers (bool): Enable/disable drawing barriers in the output
            circuit. Defaults to True.
        justify (str) : `left`, `right` or `none`. Defaults to `left`. Says how
            the circuit should be justified.
        idle_wires (bool): Include idle wires. Default is True.
        with_layout (bool): Include layout information, with labels on the physical
            layout. Default: True
        initial_state (bool): Optional. Adds |0> in the beginning of the line.
            Default: `False`.
        cregbundle (bool): Optional. If set True, bundle classical registers.
        wire_order (list): Optional. A list of integers used to reorder the display
            of the bits. The list must have an entry for every bit with the bits
            in the range 0 to (num_qubits + num_clbits).

    Returns:
        str: Latex string appropriate for writing to file.
    """
    qubits, clbits, nodes = _utils._get_layered_instructions(circuit, reverse_bits=reverse_bits, justify=justify, idle_wires=idle_wires, wire_order=wire_order)
    qcimg = _latex.QCircuitImage(qubits, clbits, nodes, scale, style=style, reverse_bits=reverse_bits, plot_barriers=plot_barriers, initial_state=initial_state, cregbundle=cregbundle, with_layout=with_layout, circuit=circuit)
    latex = qcimg.latex()
    if filename:
        with open(filename, 'w') as latex_file:
            latex_file.write(latex)
    return latex