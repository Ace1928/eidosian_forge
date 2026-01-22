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
@error(DUPLICATE_FACTORS)
def _check_duplicate_factors(self):
    dupes = [item for item, count in Counter(self.factors).items() if count > 1]
    if dupes:
        return 'duplicate factors found: %s' % ', '.join((repr(x) for x in dupes))