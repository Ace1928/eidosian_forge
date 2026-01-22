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
def _convert_to_font(cls, font_dict):
    """
        Convert ``font_dict`` to an openpyxl v2 Font object.

        Parameters
        ----------
        font_dict : dict
            A dict with zero or more of the following keys (or their synonyms).
                'name'
                'size' ('sz')
                'bold' ('b')
                'italic' ('i')
                'underline' ('u')
                'strikethrough' ('strike')
                'color'
                'vertAlign' ('vertalign')
                'charset'
                'scheme'
                'family'
                'outline'
                'shadow'
                'condense'

        Returns
        -------
        font : openpyxl.styles.Font
        """
    from openpyxl.styles import Font
    _font_key_map = {'sz': 'size', 'b': 'bold', 'i': 'italic', 'u': 'underline', 'strike': 'strikethrough', 'vertalign': 'vertAlign'}
    font_kwargs = {}
    for k, v in font_dict.items():
        k = _font_key_map.get(k, k)
        if k == 'color':
            v = cls._convert_to_color(v)
        font_kwargs[k] = v
    return Font(**font_kwargs)