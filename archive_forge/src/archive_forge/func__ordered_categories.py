from pathlib import Path
import pandas as pd
from pandas.api.types import CategoricalDtype
from the Economic Research Service of the U.S. DEPARTMENT OF AGRICULTURE.
from http://research.stlouisfed.org/fred2.
from Eisenhower to Obama.
from V. M. Savage and G. B. West. A quantitative, theoretical
def _ordered_categories(df, categories):
    """
    Make the columns in df categorical

    Parameters
    ----------
    df : pd.Dataframe
        Dataframe
    categories: dict
        Of the form {str: list},
        where the key the column name and the value is
        the ordered category list
    """
    for col, cats in categories.items():
        df[col] = df[col].astype(CategoricalDtype(cats, ordered=True))
    return df