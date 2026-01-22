import os
import stat
import sys
import typing as t
from datetime import datetime
from gettext import gettext as _
from gettext import ngettext
from ._compat import _get_argv_encoding
from ._compat import open_stream
from .exceptions import BadParameter
from .utils import format_filename
from .utils import LazyFile
from .utils import safecall
class FloatRange(_NumberRangeBase, FloatParamType):
    """Restrict a :data:`click.FLOAT` value to a range of accepted
    values. See :ref:`ranges`.

    If ``min`` or ``max`` are not passed, any value is accepted in that
    direction. If ``min_open`` or ``max_open`` are enabled, the
    corresponding boundary is not included in the range.

    If ``clamp`` is enabled, a value outside the range is clamped to the
    boundary instead of failing. This is not supported if either
    boundary is marked ``open``.

    .. versionchanged:: 8.0
        Added the ``min_open`` and ``max_open`` parameters.
    """
    name = 'float range'

    def __init__(self, min: t.Optional[float]=None, max: t.Optional[float]=None, min_open: bool=False, max_open: bool=False, clamp: bool=False) -> None:
        super().__init__(min=min, max=max, min_open=min_open, max_open=max_open, clamp=clamp)
        if (min_open or max_open) and clamp:
            raise TypeError('Clamping is not supported for open bounds.')

    def _clamp(self, bound: float, dir: 'te.Literal[1, -1]', open: bool) -> float:
        if not open:
            return bound
        raise RuntimeError('Clamping is not supported for open bounds.')