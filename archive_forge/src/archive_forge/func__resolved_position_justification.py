from __future__ import annotations
from abc import ABC
from dataclasses import asdict, dataclass, field
from functools import cached_property
from types import SimpleNamespace as NS
from typing import TYPE_CHECKING, cast
from .._utils import ensure_xy_location, get_opposite_side
from .._utils.registry import Register
from ..themes.theme import theme as Theme
@property
def _resolved_position_justification(self) -> tuple[SidePosition, float] | tuple[TupleFloat2, TupleFloat2]:
    """
        Return the final position & justification to draw the guide
        """
    pos = self.elements.position
    just_view = asdict(self.guides_elements.justification)
    if isinstance(pos, str):
        just = cast(float, just_view[pos])
        return (pos, just)
    else:
        if (just := just_view['inside']) is None:
            just = pos
        just = cast(tuple[float, float], just)
        return (pos, just)