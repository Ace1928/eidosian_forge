from collections import namedtuple
from contextlib import ExitStack, contextmanager, nullcontext
from enum import Enum, IntEnum
import functools
import importlib
import inspect
import io
import itertools
import logging
import os
import sys
import time
import weakref
from weakref import WeakKeyDictionary
import numpy as np
import matplotlib as mpl
from matplotlib import (
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_managers import ToolManager
from matplotlib.cbook import _setattr_cm
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib._enums import JoinStyle, CapStyle
def _safe_pyplot_import():
    """
    Import and return ``pyplot``, correctly setting the backend if one is
    already forced.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        current_framework = cbook._get_running_interactive_framework()
        if current_framework is None:
            raise
        backend_mapping = {'qt': 'qtagg', 'gtk3': 'gtk3agg', 'gtk4': 'gtk4agg', 'wx': 'wxagg', 'tk': 'tkagg', 'macosx': 'macosx', 'headless': 'agg'}
        backend = backend_mapping[current_framework]
        rcParams['backend'] = mpl.rcParamsOrig['backend'] = backend
        import matplotlib.pyplot as plt
    return plt