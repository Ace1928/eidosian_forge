import collections
from functools import reduce, singledispatch
from typing import (Any, Dict, Iterable, List, Optional,
import numpy as np
from ase.spectrum.dosdata import DOSData, RawDOSData, GridDOSData, Info
from ase.utils.plotting import SimplePlottingAxes
@singledispatch
def _add_to_collection(other: DOSCollection, collection: DOSCollection) -> DOSCollection:
    if isinstance(other, type(collection)):
        return type(collection)(list(collection) + list(other))
    elif isinstance(other, DOSCollection):
        raise TypeError("Only DOSCollection objects of the same type may be joined with '+'.")
    else:
        raise TypeError("DOSCollection may only be joined to DOSData or DOSCollection objects with '+'.")