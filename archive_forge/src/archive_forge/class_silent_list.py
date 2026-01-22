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
class silent_list(list):
    """
    A list with a short ``repr()``.

    This is meant to be used for a homogeneous list of artists, so that they
    don't cause long, meaningless output.

    Instead of ::

        [<matplotlib.lines.Line2D object at 0x7f5749fed3c8>,
         <matplotlib.lines.Line2D object at 0x7f5749fed4e0>,
         <matplotlib.lines.Line2D object at 0x7f5758016550>]

    one will get ::

        <a list of 3 Line2D objects>

    If ``self.type`` is None, the type name is obtained from the first item in
    the list (if any).
    """

    def __init__(self, type, seq=None):
        self.type = type
        if seq is not None:
            self.extend(seq)

    def __repr__(self):
        if self.type is not None or len(self) != 0:
            tp = self.type if self.type is not None else type(self[0]).__name__
            return f'<a list of {len(self)} {tp} objects>'
        else:
            return '<an empty list>'