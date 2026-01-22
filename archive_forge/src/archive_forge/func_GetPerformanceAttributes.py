import os
import sys
import platform
import re
import gc
import operator
import warnings
from functools import partial, wraps
import shutil
import contextlib
from tempfile import mkdtemp, mkstemp
from unittest.case import SkipTest
from warnings import WarningMessage
import pprint
import sysconfig
import numpy as np
from numpy.core import (
from numpy import isfinite, isnan, isinf
import numpy.linalg._umath_linalg
from io import StringIO
import unittest
def GetPerformanceAttributes(object, counter, instance=None, inum=-1, format=None, machine=None):
    import win32pdh
    if format is None:
        format = win32pdh.PDH_FMT_LONG
    path = win32pdh.MakeCounterPath((machine, object, instance, None, inum, counter))
    hq = win32pdh.OpenQuery()
    try:
        hc = win32pdh.AddCounter(hq, path)
        try:
            win32pdh.CollectQueryData(hq)
            type, val = win32pdh.GetFormattedCounterValue(hc, format)
            return val
        finally:
            win32pdh.RemoveCounter(hc)
    finally:
        win32pdh.CloseQuery(hq)