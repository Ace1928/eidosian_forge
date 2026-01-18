from __future__ import annotations
import io
import sys
import typing as ty
import warnings
from functools import reduce
from operator import getitem, mul
from os.path import exists, splitext
import numpy as np
from ._compression import COMPRESSED_FILE_LIKES
from .casting import OK_FLOATS, shared_range
from .externals.oset import OrderedSet
Get item from mapping, checking for dtype keys

        First do simple hash lookup, then check for a dtype key that has failed
        the hash lookup.  Look then for any known dtype keys that compare equal
        to `key`.
        