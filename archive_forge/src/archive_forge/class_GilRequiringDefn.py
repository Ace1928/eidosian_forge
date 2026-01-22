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
class GilRequiringDefn(AbstractTemplate):
    key = self.sig

    def generic(self, args, kws):
        if kws:
            raise TypeError('does not support keyword arguments')
        coerced = [actual if formal == ffi_forced_object else formal for actual, formal in zip(args, self.key.args)]
        return signature(self.key.return_type, *coerced)