from __future__ import annotations
from typing import cast, Iterable
import pandas as pd
from seaborn._core.rules import categorical_order
from typing import TYPE_CHECKING
def _reorder_columns(self, res, data):
    """Reorder result columns to match original order with new columns appended."""
    cols = [c for c in data if c in res]
    cols += [c for c in res if c not in data]
    return res.reindex(columns=pd.Index(cols))