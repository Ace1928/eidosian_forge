from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
@staticmethod
def _scale_fw_bw(scaling):
    return (lambda x: scaling * x, lambda x: x / scaling)