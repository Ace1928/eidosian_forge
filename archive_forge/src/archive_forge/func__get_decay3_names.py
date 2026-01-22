from __future__ import print_function, absolute_import, division
from collections import defaultdict, OrderedDict
from itertools import product
import math
import numpy as np
import pytest
from .. import ODESys
from ..core import integrate_auto_switch, chained_parameter_variation
from ..symbolic import SymbolicSys, ScaledSys, symmetricsys, PartiallySolvedSystem, get_logexp, _group_invariants
from ..util import requires, pycvodes_double, pycvodes_klu
from .bateman import bateman_full  # analytic, never mind the details
from .test_core import vdp_f
from . import _cetsa
def _get_decay3_names(yn, pn, **kwargs):

    def f(x, y, p):
        y = [y[n] for n in yn]
        p = [p[n] for n in pn]
        return dict(zip(yn, [-p[0] * y[0], p[0] * y[0] - p[1] * y[1], p[1] * y[1] - p[2] * y[2]]))
    return SymbolicSys.from_callback(f, names=yn, param_names=pn, dep_by_name=True, par_by_name=True, indep_name='t', **kwargs)