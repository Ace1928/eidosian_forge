import warnings
from collections import namedtuple
import operator
from . import _zeros
from ._optimize import OptimizeResult, _call_callback_maybe_halt
import numpy as np
def _update_bracket(self, c, fc):
    return _update_bracket(self.ab, self.fab, c, fc)