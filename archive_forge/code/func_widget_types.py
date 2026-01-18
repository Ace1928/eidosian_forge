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
@_staticproperty
def widget_types():
    frame = _get_frame(2)
    if not (frame.f_code.co_filename == TRAITLETS_FILE and frame.f_code.co_name in ('getmembers', 'setup_instance', 'setup_class')):
        deprecation('Widget.widget_types is deprecated.')
    return _registry