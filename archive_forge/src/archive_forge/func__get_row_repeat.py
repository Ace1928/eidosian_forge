from __future__ import annotations
from typing import (
import numpy as np
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc
import pandas as pd
from pandas.core.shared_docs import _shared_docs
from pandas.io.excel._base import BaseExcelReader
def _get_row_repeat(self, row) -> int:
    """
        Return number of times this row was repeated
        Repeating an empty row appeared to be a common way
        of representing sparse rows in the table.
        """
    from odf.namespaces import TABLENS
    return int(row.attributes.get((TABLENS, 'number-rows-repeated'), 1))