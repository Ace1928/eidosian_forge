import collections
import collections.abc
import contextlib
import functools
import gzip
import itertools
import math
import operator
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import types
import weakref
import numpy as np
import matplotlib
from matplotlib import _api, _c_internal_utils
def _g_sig_digits(value, delta):
    """
    Return the number of significant digits to %g-format *value*, assuming that
    it is known with an error of *delta*.
    """
    if delta == 0:
        delta = abs(np.spacing(value))
    return max(0, (math.floor(math.log10(abs(value))) + 1 if value else 1) - math.floor(math.log10(delta))) if math.isfinite(value) else 0