from dataclasses import dataclass
from typing import NamedTuple, Optional, Union
from pyproj.utils import is_null

        Parameters
        ----------
        other: Union["BBox", AreaOfUse]
            The other BBox to use to check.

        Returns
        -------
        bool:
            True if this BBox contains the other bbox.
        