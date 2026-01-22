from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def _is_autonomous(indep, exprs):
    """ Whether the expressions for the dependent variables are autonomous.

    Note that the system may still behave as an autonomous system on the interface
    of :meth:`integrate` due to use of pre-/post-processors.
    """
    if indep is None:
        return True
    for expr in exprs:
        try:
            in_there = indep in expr.free_symbols
        except:
            in_there = expr.has(indep)
        if in_there:
            return False
    return True