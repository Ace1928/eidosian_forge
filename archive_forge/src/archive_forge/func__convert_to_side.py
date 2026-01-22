from __future__ import annotations
import mmap
from typing import (
import numpy as np
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc
from pandas.core.shared_docs import _shared_docs
from pandas.io.excel._base import (
from pandas.io.excel._util import (
@classmethod
def _convert_to_side(cls, side_spec):
    """
        Convert ``side_spec`` to an openpyxl v2 Side object.

        Parameters
        ----------
        side_spec : str, dict
            A string specifying the border style, or a dict with zero or more
            of the following keys (or their synonyms).
                'style' ('border_style')
                'color'

        Returns
        -------
        side : openpyxl.styles.Side
        """
    from openpyxl.styles import Side
    _side_key_map = {'border_style': 'style'}
    if isinstance(side_spec, str):
        return Side(style=side_spec)
    side_kwargs = {}
    for k, v in side_spec.items():
        k = _side_key_map.get(k, k)
        if k == 'color':
            v = cls._convert_to_color(v)
        side_kwargs[k] = v
    return Side(**side_kwargs)