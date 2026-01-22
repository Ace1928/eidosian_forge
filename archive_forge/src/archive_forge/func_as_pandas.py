from typing import Any, Dict, Iterable, List, Optional, Tuple
import pandas as pd
import pyarrow as pa
from triad.collections.schema import Schema
from triad.utils.rename import normalize_names
from .._utils.registry import fugue_plugin
from .dataframe import AnyDataFrame, DataFrame, as_fugue_df
@fugue_plugin
def as_pandas(df: AnyDataFrame) -> pd.DataFrame:
    """The generic function to convert any dataframe to a Pandas DataFrame

    :param df: the object that can be recognized as a dataframe by Fugue
    :return: the Pandas DataFrame

    .. related_topics
        How to convert any dataframe to a pandas dataframe?
    """
    return as_fugue_df(df).as_pandas()