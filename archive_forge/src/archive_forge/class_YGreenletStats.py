import os
import sys
import _yappi
import pickle
import threading
import warnings
import types
import inspect
import itertools
from contextlib import contextmanager
class YGreenletStats(_YContextStats):
    _BACKEND = GREENLET
    _STAT_CLASS = YGreenletStat
    _SORT_TYPES = {'name': 0, 'id': 1, 'totaltime': 3, 'schedcount': 4, 'ttot': 3, 'scnt': 4}
    _DEFAULT_PRINT_COLUMNS = {0: ('name', 13), 1: ('id', 5), 2: ('ttot', 8), 3: ('scnt', 10)}
    _ALL_COLUMNS = ['name', 'id', 'ttot', 'scnt']