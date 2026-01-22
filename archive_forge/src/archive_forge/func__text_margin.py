from __future__ import annotations
from abc import ABC
from dataclasses import asdict, dataclass, field
from functools import cached_property
from types import SimpleNamespace as NS
from typing import TYPE_CHECKING, cast
from .._utils import ensure_xy_location, get_opposite_side
from .._utils.registry import Register
from ..themes.theme import theme as Theme
@cached_property
def _text_margin(self) -> float:
    _margin = self.theme.getp((f'legend_text_{self.guide_kind}', 'margin'))
    _loc = get_opposite_side(self.text_position)
    return _margin.get_as(_loc[0], 'pt')