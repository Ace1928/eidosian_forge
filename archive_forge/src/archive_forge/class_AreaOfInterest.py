from dataclasses import dataclass
from typing import NamedTuple, Optional, Union
from pyproj.utils import is_null
@dataclass(frozen=True)
class AreaOfInterest:
    """
    .. versionadded:: 2.3.0

    This is the area of interest for:

    - Transformations
    - Querying for CRS data.
    """
    west_lon_degree: float
    south_lat_degree: float
    east_lon_degree: float
    north_lat_degree: float

    def __post_init__(self):
        if is_null(self.west_lon_degree) or is_null(self.south_lat_degree) or is_null(self.east_lon_degree) or is_null(self.north_lat_degree):
            raise ValueError('NaN or None values are not allowed.')