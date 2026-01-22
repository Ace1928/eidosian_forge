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
class ArgConstraint(object):

    def __init__(self, dst, src, loc):
        self.dst = dst
        self.src = src
        self.loc = loc

    def __call__(self, typeinfer):
        with new_error_context('typing of argument at {0}', self.loc):
            typevars = typeinfer.typevars
            src = typevars[self.src]
            if not src.defined:
                return
            ty = src.getone()
            if isinstance(ty, types.Omitted):
                ty = typeinfer.context.resolve_value_type_prefer_literal(ty.value)
            if not ty.is_precise():
                raise TypingError('non-precise type {}'.format(ty))
            typeinfer.add_type(self.dst, ty, loc=self.loc)