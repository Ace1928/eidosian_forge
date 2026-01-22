from __future__ import annotations
import abc
import enum
import typing
import warnings
from .constants import Sizing, WHSettings
class _ContainerElementSizingFlag(enum.IntFlag):
    NONE = 0
    BOX = enum.auto()
    FLOW = enum.auto()
    FIXED = enum.auto()
    WH_WEIGHT = enum.auto()
    WH_PACK = enum.auto()
    WH_GIVEN = enum.auto()

    @property
    def reverse_flag(self) -> tuple[frozenset[Sizing], WHSettings | None]:
        """Get flag in public API format."""
        sizing: set[Sizing] = set()
        if self & self.BOX:
            sizing.add(Sizing.BOX)
        if self & self.FLOW:
            sizing.add(Sizing.FLOW)
        if self & self.FIXED:
            sizing.add(Sizing.FIXED)
        if self & self.WH_WEIGHT:
            return (frozenset(sizing), WHSettings.WEIGHT)
        if self & self.WH_PACK:
            return (frozenset(sizing), WHSettings.PACK)
        if self & self.WH_GIVEN:
            return (frozenset(sizing), WHSettings.GIVEN)
        return (frozenset(sizing), None)

    @property
    def log_string(self) -> str:
        """Get desctiprion in public API format."""
        sizing, render = self.reverse_flag
        render_string = f' {render.upper()}' if render else ''
        return '|'.join(sorted((mode.upper() for mode in sizing))) + render_string