from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas.compat._optional import import_optional_dependency

        Compute online exponentially weighted mean per column over 2D values.

        Takes the first observation as is, then computes the subsequent
        exponentially weighted mean accounting minimum periods.
        