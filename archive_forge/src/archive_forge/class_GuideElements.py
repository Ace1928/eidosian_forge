from __future__ import annotations
from abc import ABC
from dataclasses import asdict, dataclass, field
from functools import cached_property
from types import SimpleNamespace as NS
from typing import TYPE_CHECKING, cast
from .._utils import ensure_xy_location, get_opposite_side
from .._utils.registry import Register
from ..themes.theme import theme as Theme
@dataclass
class GuideElements:
    """
    Access & calculate theming for the guide
    """
    theme: Theme
    guide: guide

    def __post_init__(self):
        self.guide_kind = type(self.guide).__name__.split('_')[-1]
        self._elements_cls = GuideElements

    @cached_property
    def margin(self):
        return self.theme.getp('legend_margin')

    @cached_property
    def title(self):
        ha = self.theme.getp(('legend_title', 'ha'))
        va = self.theme.getp(('legend_title', 'va'), 'center')
        _margin = self.theme.getp(('legend_title', 'margin'))
        _loc = get_opposite_side(self.title_position)[0]
        margin = _margin.get_as(_loc, 'pt')
        top_or_bottom = self.title_position in ('top', 'bottom')
        is_blank = self.theme.T.is_blank('legend_title')
        if self.is_vertical:
            align = ha or 'left' if top_or_bottom else va
        else:
            align = ha or 'center' if top_or_bottom else va
        return NS(margin=margin, align=align, ha='center', va='baseline', is_blank=is_blank)

    @cached_property
    def text_position(self) -> SidePosition:
        raise NotImplementedError

    @cached_property
    def _text_margin(self) -> float:
        _margin = self.theme.getp((f'legend_text_{self.guide_kind}', 'margin'))
        _loc = get_opposite_side(self.text_position)
        return _margin.get_as(_loc[0], 'pt')

    @cached_property
    def title_position(self) -> SidePosition:
        if not (pos := self.theme.getp('legend_title_position')):
            pos = 'top' if self.is_vertical else 'left'
        return pos

    @cached_property
    def direction(self) -> Orientation:
        if self.guide.direction:
            return self.guide.direction
        if not (direction := self.theme.getp('legend_direction')):
            direction = 'horizontal' if self.position in ('bottom', 'top') else 'vertical'
        return direction

    @cached_property
    def position(self) -> SidePosition | TupleFloat2:
        if (guide_pos := self.guide.position) == 'inside':
            guide_pos = self._position_inside
        if guide_pos:
            return guide_pos
        if (pos := self.theme.getp('legend_position', 'right')) == 'inside':
            pos = self._position_inside
        return pos

    @cached_property
    def _position_inside(self) -> SidePosition | TupleFloat2:
        pos = self.theme.getp('legend_position_inside')
        if isinstance(pos, tuple):
            return pos
        just = self.theme.getp('legend_justification_inside', (0.5, 0.5))
        return ensure_xy_location(just)

    @cached_property
    def is_vertical(self) -> bool:
        """
        Whether the guide is vertical
        """
        return self.direction == 'vertical'

    @cached_property
    def is_horizontal(self) -> bool:
        """
        Whether the guide is horizontal
        """
        return self.direction == 'horizontal'