import os
import sys
import typing
from contextlib import contextmanager
from collections.abc import Iterable
from IPython import get_ipython
from traitlets import (
from json import loads as jsonloads, dumps as jsondumps
from .. import comm
from base64 import standard_b64encode
from .utils import deprecation, _get_frame
from .._version import __protocol_version__, __control_protocol_version__, __jupyter_widgets_base_version__
import inspect
def _buffer_list_equal(a, b):
    """Compare two lists of buffers for equality.

    Used to decide whether two sequences of buffers (memoryviews,
    bytearrays, or python 3 bytes) differ, such that a sync is needed.

    Returns True if equal, False if unequal
    """
    if len(a) != len(b):
        return False
    if a == b:
        return True
    for ia, ib in zip(a, b):
        if memoryview(ia).cast('B') != memoryview(ib).cast('B'):
            return False
    return True