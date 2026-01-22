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
@classmethod
@functools.cache
def _fix_ipython_backend2gui(cls):
    if sys.modules.get('IPython') is None:
        return
    import IPython
    ip = IPython.get_ipython()
    if not ip:
        return
    from IPython.core import pylabtools as pt
    if not hasattr(pt, 'backend2gui') or not hasattr(ip, 'enable_matplotlib'):
        return
    backend2gui_rif = {'qt': 'qt', 'gtk3': 'gtk3', 'gtk4': 'gtk4', 'wx': 'wx', 'macosx': 'osx'}.get(cls.required_interactive_framework)
    if backend2gui_rif:
        if _is_non_interactive_terminal_ipython(ip):
            ip.enable_gui(backend2gui_rif)