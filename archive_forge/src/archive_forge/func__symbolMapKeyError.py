import itertools
import logging
import operator
import os
import time
from math import isclose
from pyomo.common.fileutils import find_library
from pyomo.common.gc_manager import PauseGC
from pyomo.opt import ProblemFormat, AbstractProblemWriter, WriterFactory
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import (
from pyomo.core.base import (
import pyomo.core.base.suffix
from pyomo.repn.standard_repn import generate_standard_repn
import pyomo.core.kernel.suffix
from pyomo.core.kernel.block import IBlock
from pyomo.core.kernel.expression import IIdentityExpression
from pyomo.core.kernel.variable import IVariable
def _symbolMapKeyError(self, err, model, map, vars):
    _errors = []
    for v in vars:
        if id(v) in map:
            continue
        if v.model() is not model.model():
            _errors.append("Variable '%s' is not part of the model being written out, but appears in an expression used on this model." % (v.name,))
        else:
            _parent = v.parent_block()
            while _parent is not None and _parent is not model:
                if _parent.ctype is not model.ctype:
                    _errors.append("Variable '%s' exists within %s '%s', but is used by an active expression.  Currently variables must be reachable through a tree of active Blocks." % (v.name, _parent.ctype.__name__, _parent.name))
                if not _parent.active:
                    _errors.append("Variable '%s' exists within deactivated %s '%s', but is used by an active expression.  Currently variables must be reachable through a tree of active Blocks." % (v.name, _parent.ctype.__name__, _parent.name))
                _parent = _parent.parent_block()
    if _errors:
        for e in _errors:
            logger.error(e)
        err.args = err.args + tuple(_errors)