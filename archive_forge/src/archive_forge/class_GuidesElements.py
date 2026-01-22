from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, fields
from functools import cached_property
from typing import TYPE_CHECKING, Literal, cast
from warnings import warn
import numpy as np
import pandas as pd
from .._utils import ensure_xy_location
from .._utils.registry import Registry
from ..exceptions import PlotnineError, PlotnineWarning
from ..iapi import (
from ..mapping.aes import rename_aesthetics
from .guide import guide
@dataclass
class GuidesElements:
    """
    Theme elements used when assembling the guides object

    This class is meant to provide convenient access to all the required
    elements having worked out good defaults for the unspecified values.
    """
    theme: theme

    @cached_property
    def box(self) -> Orientation:
        """
        The direction to layout the guides
        """
        if not (box := self.theme.getp('legend_box')):
            box = 'vertical' if self.position in {'left', 'right'} else 'horizontal'
        return box

    @cached_property
    def position(self) -> LegendPosition | Literal['none']:
        if (pos := self.theme.getp('legend_position', 'right')) == 'inside':
            pos = self._position_inside
        return pos

    @cached_property
    def _position_inside(self) -> LegendPosition:
        pos = self.theme.getp('legend_position_inside')
        if isinstance(pos, tuple):
            return pos
        just = self.theme.getp('legend_justification_inside', (0.5, 0.5))
        return ensure_xy_location(just)

    @cached_property
    def box_just(self) -> TextJustification:
        if not (box_just := self.theme.getp('legend_box_just')):
            box_just = 'left' if self.position in {'left', 'right'} else 'right'
        return box_just

    @cached_property
    def box_margin(self) -> int:
        return self.theme.getp('legend_box_margin')

    @cached_property
    def spacing(self) -> float:
        return self.theme.getp('legend_spacing')

    @cached_property
    def justification(self) -> legend_justifications_view:
        if self.position == 'none':
            return legend_justifications_view()
        dim_lookup = {'left': 1, 'right': 1, 'top': 0, 'bottom': 0}

        def _lrtb(pos):
            just = self.theme.getp(f'legend_justification_{pos}')
            idx = dim_lookup[pos]
            if just is None:
                just = (0.5, 0.5)
            elif just in VALID_JUSTIFICATION_WORDS:
                just = ensure_xy_location(just)
            elif isinstance(just, (float, int)):
                just = (just, just)
            return just[idx]

        def _inside():
            just = self.theme.getp('legend_justification_inside')
            if just is None:
                return None
            return ensure_xy_location(just)
        return legend_justifications_view(left=_lrtb('left'), right=_lrtb('right'), top=_lrtb('top'), bottom=_lrtb('bottom'), inside=_inside())