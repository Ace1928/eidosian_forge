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
def _show_traceback(method):
    """decorator for showing tracebacks"""

    def m(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as e:
            ip = get_ipython()
            if ip is None:
                self.log.warning('Exception in widget method %s: %s', method, e, exc_info=True)
            else:
                ip.showtraceback()
    return m