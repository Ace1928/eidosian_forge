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
class PrintConstraint(object):

    def __init__(self, args, vararg, loc):
        self.args = args
        self.vararg = vararg
        self.loc = loc

    def __call__(self, typeinfer):
        typevars = typeinfer.typevars
        r = fold_arg_vars(typevars, self.args, self.vararg, {})
        if r is None:
            return
        pos_args, kw_args = r
        fnty = typeinfer.context.resolve_value_type(print)
        assert fnty is not None
        sig = typeinfer.resolve_call(fnty, pos_args, kw_args)
        self.signature = sig

    def get_call_signature(self):
        return self.signature