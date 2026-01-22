import json
import struct
import zlib
import warnings
from io import BytesIO
import numpy as np
import symengine as sym
from symengine.lib.symengine_wrapper import (  # pylint: disable = no-name-in-module
from qiskit.exceptions import QiskitError
from qiskit.pulse import library, channels, instructions
from qiskit.pulse.schedule import ScheduleBlock
from qiskit.qpy import formats, common, type_keys
from qiskit.qpy.binary_io import value
from qiskit.qpy.exceptions import QpyError
from qiskit.pulse.configuration import Kernel, Discriminator
def _dumps_symbolic_expr(expr, use_symengine):
    if expr is None:
        return b''
    if use_symengine:
        expr_bytes = expr.__reduce__()[1][0]
    else:
        from sympy import srepr, sympify
        expr_bytes = srepr(sympify(expr)).encode(common.ENCODE)
    return zlib.compress(expr_bytes)