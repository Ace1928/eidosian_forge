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
class SetAttrConstraint(object):

    def __init__(self, target, attr, value, loc):
        self.target = target
        self.attr = attr
        self.value = value
        self.loc = loc

    def __call__(self, typeinfer):
        with new_error_context('typing of set attribute {0!r} at {1}', self.attr, self.loc):
            typevars = typeinfer.typevars
            if not all((typevars[var.name].defined for var in (self.target, self.value))):
                return
            targetty = typevars[self.target.name].getone()
            valty = typevars[self.value.name].getone()
            sig = typeinfer.context.resolve_setattr(targetty, self.attr, valty)
            if sig is None:
                raise TypingError('Cannot resolve setattr: (%s).%s = %s' % (targetty, self.attr, valty), loc=self.loc)
            self.signature = sig

    def get_call_signature(self):
        return self.signature