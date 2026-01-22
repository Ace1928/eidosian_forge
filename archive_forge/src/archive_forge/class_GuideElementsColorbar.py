from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from functools import cached_property
from types import SimpleNamespace as NS
from typing import TYPE_CHECKING, cast
from warnings import warn
import numpy as np
import pandas as pd
from mizani.bounds import rescale
from .._utils import get_opposite_side
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping.aes import rename_aesthetics
from ..scales.scale_continuous import scale_continuous
from .guide import GuideElements, guide
class GuideElementsColorbar(GuideElements):
    """
    Access & calculate theming for the colobar
    """

    @cached_property
    def text(self):
        size = self.theme.getp(('legend_text_colorbar', 'size'))
        ha = self.theme.getp(('legend_text_colorbar', 'ha'))
        va = self.theme.getp(('legend_text_colorbar', 'va'))
        is_blank = self.theme.T.is_blank('legend_text_colorbar')
        _loc = get_opposite_side(self.text_position)
        if self.is_vertical:
            ha = ha or _loc
            va = va or 'center'
        else:
            va = va or _loc
            ha = ha or 'center'
        return NS(margin=self._text_margin, align=None, fontsize=size, ha=ha, va=va, is_blank=is_blank)

    @cached_property
    def text_position(self) -> SidePosition:
        if not (position := self.theme.getp('legend_text_position')):
            position = 'right' if self.is_vertical else 'bottom'
        if self.is_vertical and position not in ('right', 'left'):
            msg = 'The text position for a vertical legend must be either left or right.'
            raise PlotnineError(msg)
        elif self.is_horizontal and position not in ('bottom', 'top'):
            msg = 'The text position for a horizonta legend must be either top or bottom.'
            raise PlotnineError(msg)
        return position

    @cached_property
    def key_width(self):
        dim = self.is_vertical and 'width' or 'height'
        legend_key_dim = f'legend_key_{dim}'
        inherited = self.theme.T.get(legend_key_dim) is None
        scale = 1.45 if inherited else 1
        return np.round(self.theme.getp(legend_key_dim) * scale)

    @cached_property
    def key_height(self):
        dim = self.is_vertical and 'height' or 'width'
        legend_key_dim = f'legend_key_{dim}'
        inherited = self.theme.T.get(legend_key_dim) is None
        scale = 1.45 * 5 if inherited else 1
        return np.round(self.theme.getp(legend_key_dim) * scale)

    @cached_property
    def frame(self):
        lw = self.theme.getp(('legend_frame', 'linewidth'), 0)
        return NS(linewidth=lw)

    @cached_property
    def ticks_length(self):
        return self.theme.getp('legend_ticks_length')

    @cached_property
    def ticks(self):
        lw = self.theme.getp(('legend_ticks', 'linewidth'))
        return NS(linewidth=lw)