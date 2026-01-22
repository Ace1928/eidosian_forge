from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
@property
def _caption_macro(self) -> str:
    """Caption macro, extracted from self.caption.

        With short caption:
            \\caption[short_caption]{caption_string}.

        Without short caption:
            \\caption{caption_string}.
        """
    if self.caption:
        return ''.join(['\\caption', f'[{self.short_caption}]' if self.short_caption else '', f'{{{self.caption}}}'])
    return ''