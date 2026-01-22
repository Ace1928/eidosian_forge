import os
import textwrap
from io import StringIO
from sympy import __version__ as sympy_version
from sympy.core import Symbol, S, Tuple, Equality, Function, Basic
from sympy.printing.c import c_code_printers
from sympy.printing.codeprinter import AssignmentError
from sympy.printing.fortran import FCodePrinter
from sympy.printing.julia import JuliaCodePrinter
from sympy.printing.octave import OctaveCodePrinter
from sympy.printing.rust import RustCodePrinter
from sympy.tensor import Idx, Indexed, IndexedBase
from sympy.matrices import (MatrixSymbol, ImmutableMatrix, MatrixBase,
from sympy.utilities.iterables import is_sequence
def _printer_method_with_settings(self, method, settings=None, *args, **kwargs):
    settings = settings or {}
    ori = {k: self.printer._settings[k] for k in settings}
    for k, v in settings.items():
        self.printer._settings[k] = v
    result = getattr(self.printer, method)(*args, **kwargs)
    for k, v in ori.items():
        self.printer._settings[k] = v
    return result