import logging
import operator
import warnings
from functools import reduce
from copy import copy
from pprint import pformat
from collections import defaultdict
from numba import config
from numba.core import ir, ir_utils, errors
from numba.core.analysis import compute_cfg_from_blocks
class _FreshVarHandler(_BaseHandler):
    """Replaces assignment target with new fresh variables.
    """

    def on_assign(self, states, assign):
        if assign.target.name == states['varname']:
            scope = states['scope']
            defmap = states['defmap']
            if len(defmap) == 0:
                newtarget = assign.target
                _logger.debug('first assign: %s', newtarget)
                if newtarget.name not in scope.localvars:
                    wmsg = f'variable {newtarget.name!r} is not in scope.'
                    warnings.warn(errors.NumbaIRAssumptionWarning(wmsg, loc=assign.loc))
            else:
                newtarget = scope.redefine(assign.target.name, loc=assign.loc)
            assign = ir.Assign(target=newtarget, value=assign.value, loc=assign.loc)
            defmap[states['label']].append(assign)
        return assign

    def on_other(self, states, stmt):
        return stmt