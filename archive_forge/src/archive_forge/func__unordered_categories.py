from pathlib import Path
import pandas as pd
from pandas.api.types import CategoricalDtype
from the Economic Research Service of the U.S. DEPARTMENT OF AGRICULTURE.
from http://research.stlouisfed.org/fred2.
from Eisenhower to Obama.
from V. M. Savage and G. B. West. A quantitative, theoretical
def _unordered_categories(df, columns):
    """Make the columns in df categorical"""
    for col in columns:
        df[col] = df[col].astype(CategoricalDtype(ordered=False))
    return df