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
class JSONFormatter(BaseFormatter):
    """A JSON string formatter.

    To define the callables that compute the JSONable representation of
    your objects, define a :meth:`_repr_json_` method or use the :meth:`for_type`
    or :meth:`for_type_by_name` methods to register functions that handle
    this.

    The return value of this formatter should be a JSONable list or dict.
    JSON scalars (None, number, string) are not allowed, only dict or list containers.
    """
    format_type = Unicode('application/json')
    _return_type = (list, dict)
    print_method = ObjectName('_repr_json_')

    def _check_return(self, r, obj):
        """Check that a return value is appropriate

        Return the value if so, None otherwise, warning if invalid.
        """
        if r is None:
            return
        md = None
        if isinstance(r, tuple):
            r, md = r
        assert not isinstance(r, str), 'JSON-as-string has been deprecated since IPython < 3'
        if md is not None:
            r = (r, md)
        return super(JSONFormatter, self)._check_return(r, obj)