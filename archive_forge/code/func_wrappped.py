import re
import warnings
from numba.core import typing, sigutils
from numba.pycc.compiler import ExportEntry
def wrappped(func):
    fn_argtys, fn_retty = sigutils.normalize_signature(sig)
    signature = typing.signature(fn_retty, *fn_argtys)
    entry = ExportEntry(symbol=sym, signature=signature, function=func)
    export_registry.append(entry)