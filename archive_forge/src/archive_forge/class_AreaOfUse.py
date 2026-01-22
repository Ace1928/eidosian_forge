from dataclasses import dataclass
from typing import NamedTuple, Optional, Union
from pyproj.utils import is_null
class AreaOfUse(NamedTuple):
    """
    .. versionadded:: 2.0.0

    Area of Use for CRS, CoordinateOperation, or a Transformer.
    """
    west: float
    south: float
    east: float
    north: float
    name: Optional[str] = None

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """
        The bounds of the area of use.

        Returns
        -------
        tuple[float, float, float, float]
            west, south, east, and north bounds.
        """
        return (self.west, self.south, self.east, self.north)

    def __str__(self) -> str:
        return f'- name: {self.name}\n- bounds: {self.bounds}'