from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def _get_analytic_stiffness_cb(self):
    J = self.get_jac()
    eig_vals = list(J.eigenvals().keys())
    return self._callback_factory(eig_vals)