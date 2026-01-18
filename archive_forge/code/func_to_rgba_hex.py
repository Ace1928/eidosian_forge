from __future__ import annotations
import inspect
import itertools
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import suppress
from copy import deepcopy
from dataclasses import field
from typing import TYPE_CHECKING, cast, overload
from warnings import warn
import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping import aes
def to_rgba_hex(c: ColorType, a: float) -> str:
    """
        Conver rgb color to rgba hex value

        If color c has an alpha channel, then alpha value
        a is ignored
        """
    from matplotlib.colors import colorConverter, to_hex
    if c in ('None', 'none'):
        return c
    _has_alpha = has_alpha(c)
    c = to_hex(c, keep_alpha=_has_alpha)
    if not _has_alpha:
        arr = colorConverter.to_rgba(c, a)
        return to_hex(arr, keep_alpha=True)
    return c