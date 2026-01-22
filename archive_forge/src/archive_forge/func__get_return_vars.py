import logging
import operator
import contextlib
import itertools
from pprint import pprint
from collections import OrderedDict, defaultdict
from functools import reduce
from numba.core import types, utils, typing, ir, config
from numba.core.typing.templates import Signature
from numba.core.errors import (TypingError, UntypedAttributeError,
from numba.core.funcdesc import qualifying_prefix
from numba.core.typeconv import Conversion
def _get_return_vars(self):
    rets = []
    for blk in self.blocks.values():
        inst = blk.terminator
        if isinstance(inst, ir.Return):
            rets.append(inst.value)
    return rets