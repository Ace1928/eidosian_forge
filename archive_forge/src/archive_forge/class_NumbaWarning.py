import abc
import contextlib
import os
import sys
import warnings
import numba.core.config
import numpy as np
from collections import defaultdict
from functools import wraps
from abc import abstractmethod
class NumbaWarning(Warning):
    """
    Base category for all Numba compiler warnings.
    """

    def __init__(self, msg, loc=None, highlighting=True):
        self.msg = msg
        self.loc = loc
        if highlighting and _is_numba_core_config_loaded():
            highlight = termcolor().errmsg
        else:

            def highlight(x):
                return x
        if loc:
            super(NumbaWarning, self).__init__(highlight('%s\n%s\n' % (msg, loc.strformat())))
        else:
            super(NumbaWarning, self).__init__(highlight('%s' % (msg,)))