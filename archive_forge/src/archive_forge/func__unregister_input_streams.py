import sys
import weakref
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from itertools import groupby
from numbers import Number
from types import FunctionType
import numpy as np
import pandas as pd
import param
from packaging.version import Version
from .core import util
from .core.ndmapping import UniformNdMapping
def _unregister_input_streams(self):
    """
        Unregister callbacks on input streams and clear input streams list
        """
    for stream in self.input_streams:
        stream.source = None
        stream.clear()
    self.input_streams.clear()