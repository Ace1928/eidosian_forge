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
def _convert_to_border(cls, border_dict):
    """
        Convert ``border_dict`` to an openpyxl v2 Border object.

        Parameters
        ----------
        border_dict : dict
            A dict with zero or more of the following keys (or their synonyms).
                'left'
                'right'
                'top'
                'bottom'
                'diagonal'
                'diagonal_direction'
                'vertical'
                'horizontal'
                'diagonalUp' ('diagonalup')
                'diagonalDown' ('diagonaldown')
                'outline'

        Returns
        -------
        border : openpyxl.styles.Border
        """
    from openpyxl.styles import Border
    _border_key_map = {'diagonalup': 'diagonalUp', 'diagonaldown': 'diagonalDown'}
    border_kwargs = {}
    for k, v in border_dict.items():
        k = _border_key_map.get(k, k)
        if k == 'color':
            v = cls._convert_to_color(v)
        if k in ['left', 'right', 'top', 'bottom', 'diagonal']:
            v = cls._convert_to_side(v)
        border_kwargs[k] = v
    return Border(**border_kwargs)