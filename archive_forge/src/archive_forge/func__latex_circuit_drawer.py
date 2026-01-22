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
@_optionals.HAS_PDFLATEX.require_in_call('LaTeX circuit drawing')
@_optionals.HAS_PDFTOCAIRO.require_in_call('LaTeX circuit drawing')
@_optionals.HAS_PIL.require_in_call('LaTeX circuit drawing')
def _latex_circuit_drawer(circuit, scale=0.7, style=None, filename=None, plot_barriers=True, reverse_bits=False, justify=None, idle_wires=True, with_layout=True, initial_state=False, cregbundle=None, wire_order=None):
    """Draw a quantum circuit based on latex (Qcircuit package)

    Requires version >=2.6.0 of the qcircuit LaTeX package.

    Args:
        circuit (QuantumCircuit): a quantum circuit
        scale (float): scaling factor
        style (dict or str): dictionary of style or file name of style file
        filename (str): file path to save image to
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
        cregbundle (bool): Optional. If set True, bundle classical registers.  On by default, if
            this is possible for the given circuit, otherwise off.
        wire_order (list): Optional. A list of integers used to reorder the display
            of the bits. The list must have an entry for every bit with the bits
            in the range 0 to (num_qubits + num_clbits).

    Returns:
        PIL.Image: an in-memory representation of the circuit diagram

    Raises:
        MissingOptionalLibraryError: if pillow, pdflatex, or poppler are not installed
        VisualizationError: if one of the conversion utilities failed for some internal or
            file-access reason.
    """
    from PIL import Image
    tmpfilename = 'circuit'
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmppath = os.path.join(tmpdirname, tmpfilename + '.tex')
        _generate_latex_source(circuit, filename=tmppath, scale=scale, style=style, plot_barriers=plot_barriers, reverse_bits=reverse_bits, justify=justify, idle_wires=idle_wires, with_layout=with_layout, initial_state=initial_state, cregbundle=cregbundle, wire_order=wire_order)
        try:
            subprocess.run(['pdflatex', '-halt-on-error', f'-output-directory={tmpdirname}', f'{tmpfilename + '.tex'}'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
        except OSError as exc:
            raise VisualizationError('`pdflatex` command could not be run.') from exc
        except subprocess.CalledProcessError as exc:
            with open('latex_error.log', 'wb') as error_file:
                error_file.write(exc.stdout)
            logger.warning('Unable to compile LaTeX. Perhaps you are missing the `qcircuit` package. The output from the `pdflatex` command is in `latex_error.log`.')
            raise VisualizationError('`pdflatex` call did not succeed: see `latex_error.log`.') from exc
        base = os.path.join(tmpdirname, tmpfilename)
        try:
            subprocess.run(['pdftocairo', '-singlefile', '-png', '-q', base + '.pdf', base], check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            message = '`pdftocairo` failed to produce an image.'
            logger.warning(message)
            raise VisualizationError(message) from exc
        image = Image.open(base + '.png')
        image = trim_image(image)
        if filename:
            if filename.endswith('.pdf'):
                shutil.move(base + '.pdf', filename)
            else:
                try:
                    image.save(filename)
                except (ValueError, OSError) as exc:
                    raise VisualizationError(f"Pillow could not write the image file '{filename}'.") from exc
        return image