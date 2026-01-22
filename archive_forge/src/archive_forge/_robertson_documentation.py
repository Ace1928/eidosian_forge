from __future__ import division, print_function, absolute_import
import math
import numpy as np
from ..util import import_
from ..core import RecoverableError
from ..symbolic import ScaledSys

    reduced:
    0: A, B, C
    1: B, C
    2: A, C
    3: A, B
    