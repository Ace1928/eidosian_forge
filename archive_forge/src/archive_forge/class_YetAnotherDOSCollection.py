import pytest
from typing import Iterable
import numpy as np
from ase.spectrum.doscollection import (DOSCollection,
from ase.spectrum.dosdata import DOSData, RawDOSData, GridDOSData
class YetAnotherDOSCollection(DOSCollection):
    """Inherit from abstract base class to check its features"""

    def __init__(self, dos_series: Iterable[DOSData]) -> None:
        super().__init__(dos_series)