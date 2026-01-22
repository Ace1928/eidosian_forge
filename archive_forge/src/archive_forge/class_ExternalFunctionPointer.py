import traceback
from collections import namedtuple, defaultdict
import itertools
import logging
import textwrap
from shutil import get_terminal_size
from .abstract import Callable, DTypeSpec, Dummy, Literal, Type, weakref
from .common import Opaque
from .misc import unliteral
from numba.core import errors, utils, types, config
from numba.core.typeconv import Conversion
class ExternalFunctionPointer(BaseFunction):
    """
    A pointer to a native function (e.g. exported via ctypes or cffi).
    *get_pointer* is a Python function taking an object
    and returning the raw pointer value as an int.
    """

    def __init__(self, sig, get_pointer, cconv=None):
        from numba.core.typing.templates import AbstractTemplate, make_concrete_template, signature
        from numba.core.types import ffi_forced_object
        if sig.return_type == ffi_forced_object:
            raise TypeError('Cannot return a pyobject from a external function')
        self.sig = sig
        self.requires_gil = any((a == ffi_forced_object for a in self.sig.args))
        self.get_pointer = get_pointer
        self.cconv = cconv
        if self.requires_gil:

            class GilRequiringDefn(AbstractTemplate):
                key = self.sig

                def generic(self, args, kws):
                    if kws:
                        raise TypeError('does not support keyword arguments')
                    coerced = [actual if formal == ffi_forced_object else formal for actual, formal in zip(args, self.key.args)]
                    return signature(self.key.return_type, *coerced)
            template = GilRequiringDefn
        else:
            template = make_concrete_template('CFuncPtr', sig, [sig])
        super(ExternalFunctionPointer, self).__init__(template)

    @property
    def key(self):
        return (self.sig, self.cconv, self.get_pointer)