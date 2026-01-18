from __future__ import annotations
from statsmodels.compat.python import lmap
from functools import reduce
import numpy as np
from pandas import DataFrame, Series, isnull, MultiIndex
import statsmodels.tools.data as data_util
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import MissingDataError
@cache_writable()
def xnames(self) -> list[str] | None:
    exog = self.orig_exog
    if exog is not None:
        xnames = self._get_names(exog)
        if not xnames:
            xnames = _make_exog_names(self.exog)
        return list(xnames)
    return None