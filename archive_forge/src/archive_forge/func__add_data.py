import collections
from functools import reduce, singledispatch
from typing import (Any, Dict, Iterable, List, Optional,
import numpy as np
from ase.spectrum.dosdata import DOSData, RawDOSData, GridDOSData, Info
from ase.utils.plotting import SimplePlottingAxes
@_add_to_collection.register(DOSData)
def _add_data(other: DOSData, collection: DOSCollection) -> DOSCollection:
    """Return a new DOSCollection with an additional DOSData item"""
    return type(collection)(list(collection) + [other])