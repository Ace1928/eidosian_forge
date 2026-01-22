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
def _convert_to_style_kwargs(cls, style_dict: dict) -> dict[str, Serialisable]:
    """
        Convert a style_dict to a set of kwargs suitable for initializing
        or updating-on-copy an openpyxl v2 style object.

        Parameters
        ----------
        style_dict : dict
            A dict with zero or more of the following keys (or their synonyms).
                'font'
                'fill'
                'border' ('borders')
                'alignment'
                'number_format'
                'protection'

        Returns
        -------
        style_kwargs : dict
            A dict with the same, normalized keys as ``style_dict`` but each
            value has been replaced with a native openpyxl style object of the
            appropriate class.
        """
    _style_key_map = {'borders': 'border'}
    style_kwargs: dict[str, Serialisable] = {}
    for k, v in style_dict.items():
        k = _style_key_map.get(k, k)
        _conv_to_x = getattr(cls, f'_convert_to_{k}', lambda x: None)
        new_v = _conv_to_x(v)
        if new_v:
            style_kwargs[k] = new_v
    return style_kwargs