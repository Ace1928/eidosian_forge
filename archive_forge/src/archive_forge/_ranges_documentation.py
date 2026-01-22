from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas._libs.lib import i8max
from pandas._libs.tslibs import (

    A special case for _generate_range_overflow_safe where `periods * stride`
    can be calculated without overflowing int64 bounds.
    