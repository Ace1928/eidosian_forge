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
class BuildTupleConstraint(object):

    def __init__(self, target, items, loc):
        self.target = target
        self.items = items
        self.loc = loc

    def __call__(self, typeinfer):
        with new_error_context('typing of tuple at {0}', self.loc):
            typevars = typeinfer.typevars
            tsets = [typevars[i.name].get() for i in self.items]
            for vals in itertools.product(*tsets):
                if vals and all((vals[0] == v for v in vals)):
                    tup = types.UniTuple(dtype=vals[0], count=len(vals))
                else:
                    tup = types.Tuple(vals)
                assert tup.is_precise()
                typeinfer.add_type(self.target, tup, loc=self.loc)