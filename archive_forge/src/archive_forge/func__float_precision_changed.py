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
@observe('float_precision')
def _float_precision_changed(self, change):
    """float_precision changed, set float_format accordingly.

        float_precision can be set by int or str.
        This will set float_format, after interpreting input.
        If numpy has been imported, numpy print precision will also be set.

        integer `n` sets format to '%.nf', otherwise, format set directly.

        An empty string returns to defaults (repr for float, 8 for numpy).

        This parameter can be set via the '%precision' magic.
        """
    new = change['new']
    if '%' in new:
        fmt = new
        try:
            fmt % 3.14159
        except Exception as e:
            raise ValueError('Precision must be int or format string, not %r' % new) from e
    elif new:
        try:
            i = int(new)
            assert i >= 0
        except ValueError as e:
            raise ValueError('Precision must be int or format string, not %r' % new) from e
        except AssertionError as e:
            raise ValueError('int precision must be non-negative, not %r' % i) from e
        fmt = '%%.%if' % i
        if 'numpy' in sys.modules:
            import numpy
            numpy.set_printoptions(precision=i)
    else:
        fmt = '%r'
        if 'numpy' in sys.modules:
            import numpy
            numpy.set_printoptions(precision=8)
    self.float_format = fmt