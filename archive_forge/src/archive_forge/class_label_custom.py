from __future__ import annotations
import re
import typing
from bisect import bisect_right
from dataclasses import dataclass
from zoneinfo import ZoneInfo
import numpy as np
from .breaks import timedelta_helper
from .utils import (
@dataclass
class label_custom:
    """
    Creating a custom labelling function

    Parameters
    ----------
    fmt : str, optional
        Format string. Default is the generic new style
        format braces, ``{}``.
    style : 'new' | 'old'
        Whether to use new style or old style formatting.
        New style uses the :meth:`str.format` while old
        style uses ``%``. The format string must be written
        accordingly.

    Examples
    --------
    >>> label = label_custom('{:.2f} USD')
    >>> label([3.987, 2, 42.42])
    ['3.99 USD', '2.00 USD', '42.42 USD']
    """
    fmt: str = '{}'
    style: Literal['old', 'new'] = 'new'

    def __call__(self, x: FloatArrayLike) -> Sequence[str]:
        """
        Format a sequence of inputs

        Parameters
        ----------
        x : array
            Input

        Returns
        -------
        out : list
            List of strings.
        """
        if self.style == 'new':
            return [self.fmt.format(val) for val in x]
        elif self.style == 'old':
            return [self.fmt % val for val in x]
        else:
            raise ValueError("style should be either 'new' or 'old'")