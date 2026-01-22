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
def _register_input_stream(self):
    """
        Register callback on input_stream to watch for changes
        """

    def perform_update(**kwargs):
        self.values.append(kwargs)
        self.event()
    self.input_stream.add_subscriber(perform_update)