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
def _unify_return_types(self, rettypes):
    if rettypes:
        unified = self.context.unify_types(*rettypes)
        if isinstance(unified, types.FunctionType):
            return unified
        if unified is None or not unified.is_precise():

            def check_type(atype):
                lst = []
                for k, v in self.typevars.items():
                    if atype == v.type:
                        lst.append(k)
                returns = {}
                for x in reversed(lst):
                    for block in self.func_ir.blocks.values():
                        for instr in block.find_insts(ir.Return):
                            value = instr.value
                            if isinstance(value, ir.Var):
                                name = value.name
                            else:
                                pass
                            if x == name:
                                returns[x] = instr
                                break
                interped = ''
                for name, offender in returns.items():
                    loc = getattr(offender, 'loc', ir.unknown_loc)
                    msg = "Return of: IR name '%s', type '%s', location: %s"
                    interped = msg % (name, atype, loc.strformat())
                return interped
            problem_str = []
            for xtype in rettypes:
                problem_str.append(_termcolor.errmsg(check_type(xtype)))
            raise TypingError("Can't unify return type from the following types: %s" % ', '.join(sorted(map(str, rettypes))) + '\n' + '\n'.join(problem_str))
        return unified
    else:
        return types.none