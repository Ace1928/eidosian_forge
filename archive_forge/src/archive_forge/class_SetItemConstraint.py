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
class SetItemConstraint(SetItemRefinement):

    def __init__(self, target, index, value, loc):
        self.target = target
        self.index = index
        self.value = value
        self.loc = loc

    def __call__(self, typeinfer):
        with new_error_context('typing of setitem at {0}', self.loc):
            typevars = typeinfer.typevars
            if not all((typevars[var.name].defined for var in (self.target, self.index, self.value))):
                return
            targetty = typevars[self.target.name].getone()
            idxty = typevars[self.index.name].getone()
            valty = typevars[self.value.name].getone()
            sig = typeinfer.context.resolve_setitem(targetty, idxty, valty)
            if sig is None:
                raise TypingError('Cannot resolve setitem: %s[%s] = %s' % (targetty, idxty, valty), loc=self.loc)
            self.signature = sig
            self._refine_target_type(typeinfer, targetty, idxty, valty, sig)

    def get_call_signature(self):
        return self.signature