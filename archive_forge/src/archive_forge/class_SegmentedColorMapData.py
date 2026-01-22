from __future__ import annotations
import typing
class SegmentedColorMapData(TypedDict):
    red: Sequence[tuple[float, float, float]]
    green: Sequence[tuple[float, float, float]]
    blue: Sequence[tuple[float, float, float]]