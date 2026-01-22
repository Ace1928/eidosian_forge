from collections.abc import Sequence
import functools
import inspect
import linecache
import pydoc
import sys
import time
import traceback
import types
from types import TracebackType
from typing import Any, List, Optional, Tuple
import stack_data
from pygments.formatters.terminal256 import Terminal256Formatter
from pygments.styles import get_style_by_name
import IPython.utils.colorable as colorable
from IPython import get_ipython
from IPython.core import debugger
from IPython.core.display_trap import DisplayTrap
from IPython.core.excolors import exception_colors
from IPython.utils import PyColorize
from IPython.utils import path as util_path
from IPython.utils import py3compat
from IPython.utils.terminal import get_terminal_size
@classmethod
def _from_stack_data_FrameInfo(cls, frame_info):
    return cls(getattr(frame_info, 'description', None), getattr(frame_info, 'filename', None), getattr(frame_info, 'lineno', None), getattr(frame_info, 'frame', None), getattr(frame_info, 'code', None), sd=frame_info, context=None)