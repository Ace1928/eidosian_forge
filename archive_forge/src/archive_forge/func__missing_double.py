from __future__ import annotations
from collections import abc
from datetime import datetime
import struct
from typing import TYPE_CHECKING
import warnings
import numpy as np
from pandas.util._decorators import Appender
from pandas.util._exceptions import find_stack_level
import pandas as pd
from pandas.io.common import get_handle
from pandas.io.sas.sasreader import ReaderBase
def _missing_double(self, vec):
    v = vec.view(dtype='u1,u1,u2,u4')
    miss = (v['f1'] == 0) & (v['f2'] == 0) & (v['f3'] == 0)
    miss1 = (v['f0'] >= 65) & (v['f0'] <= 90) | (v['f0'] == 95) | (v['f0'] == 46)
    miss &= miss1
    return miss