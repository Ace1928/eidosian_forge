from __future__ import absolute_import
import types
import warnings
from autograd.extend import primitive, notrace_primitive
import numpy as _np
import autograd.builtins as builtins
from numpy.core.einsumfunc import _parse_einsum_input
def wrap_intdtype(cls):

    class IntdtypeSubclass(cls):
        __new__ = notrace_primitive(cls.__new__)
    return IntdtypeSubclass