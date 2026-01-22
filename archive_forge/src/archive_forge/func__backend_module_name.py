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
def _backend_module_name(name):
    """
    Convert a backend name (either a standard backend -- "Agg", "TkAgg", ... --
    or a custom backend -- "module://...") to the corresponding module name).
    """
    return name[9:] if name.startswith('module://') else f'matplotlib.backends.backend_{name.lower()}'