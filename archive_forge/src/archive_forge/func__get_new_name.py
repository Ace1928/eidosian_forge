from enum import Enum
from typing import Dict, List, Sequence, Tuple, cast
import numpy as np
import pandas
from pandas._typing import IndexLabel
from pandas.api.types import is_scalar
def _get_new_name(col: IndexLabel, suffix: str) -> IndexLabel:
    if col in conflicting_cols:
        return (f'{col[0]}{suffix}', *col[1:]) if isinstance(col, tuple) else f'{col}{suffix}'
    else:
        return col