from __future__ import (absolute_import, division, print_function)
from functools import reduce
import inspect
import math
import operator
import sys
from pkg_resources import parse_requirements, parse_version
import numpy as np
import pytest
def transform_exprs_dep(fw, bw, dep_exprs, check=True):
    """ Transform y[:] in dydx

    Parameters
    ----------
    fw: expression
        forward transformation
    bw: expression
        backward transformation
    dep_exprs: iterable of (symbol, expression) pairs
        pairs of (dependent variable, derivative expressions),
        i.e. (y, dydx) pairs
    check: bool (default: True)
        whether to verification of the analytic correctness should
        be performed

    Returns
    -------
    List of transformed expressions for dydx

    """
    if len(fw) != len(dep_exprs) or len(fw) != len(bw):
        raise ValueError('Incompatible lengths')
    dep, exprs = zip(*dep_exprs)
    if check:
        check_transforms(fw, bw, dep)
    bw_subs = list(zip(dep, bw))
    return [(e * f.diff(y)).subs(bw_subs) for f, y, e in zip(fw, dep, exprs)]