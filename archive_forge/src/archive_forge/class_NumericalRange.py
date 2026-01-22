from __future__ import annotations
import logging # isort:skip
from collections import Counter
from math import nan
from ..core.enums import PaddingUnits, StartEnd
from ..core.has_props import abstract
from ..core.properties import (
from ..core.validation import error
from ..core.validation.errors import DUPLICATE_FACTORS
from ..model import Model
@abstract
class NumericalRange(Range):
    """ A base class for numerical ranges.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    start = Required(Either(Float, Datetime, TimeDelta), help='\n    The start of the range.\n    ')
    end = Required(Either(Float, Datetime, TimeDelta), help='\n    The end of the range.\n    ')