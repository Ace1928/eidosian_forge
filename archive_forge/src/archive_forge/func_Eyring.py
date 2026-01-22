from __future__ import (absolute_import, division, print_function)
import math
from collections import OrderedDict
from ..util import import_
import numpy as np
from ..symbolic import SymbolicSys, TransformedSys, symmetricsys
def Eyring(dH, dS, T, R, kB_over_h, be):
    return kB_over_h * T * be.exp(-(dH - T * dS) / (R * T))