import abc
import sys
import traceback
import warnings
from io import StringIO
from decorator import decorator
from traitlets.config.configurable import Configurable
from .getipython import get_ipython
from ..utils.sentinel import Sentinel
from ..utils.dir2 import get_real_method
from ..lib import pretty
from traitlets import (
from typing import Any
class FormatterABC(metaclass=abc.ABCMeta):
    """ Abstract base class for Formatters.

    A formatter is a callable class that is responsible for computing the
    raw format data for a particular format type (MIME type). For example,
    an HTML formatter would have a format type of `text/html` and would return
    the HTML representation of the object when called.
    """
    format_type = 'text/plain'
    enabled = True

    @abc.abstractmethod
    def __call__(self, obj):
        """Return a JSON'able representation of the object.

        If the object cannot be formatted by this formatter,
        warn and return None.
        """
        return repr(obj)