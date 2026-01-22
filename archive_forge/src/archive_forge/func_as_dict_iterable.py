from typing import Any, Dict, Iterable, List, Optional, Tuple
import pandas as pd
import pyarrow as pa
from triad.collections.schema import Schema
from triad.utils.rename import normalize_names
from .._utils.registry import fugue_plugin
from .dataframe import AnyDataFrame, DataFrame, as_fugue_df
@fugue_plugin
def as_dict_iterable(df: AnyDataFrame, columns: Optional[List[str]]=None) -> Iterable[Dict[str, Any]]:
    """Convert any dataframe to iterable of python dicts

    :param df: the object that can be recognized as a dataframe by Fugue
    :param columns: columns to extract, defaults to None
    :return: iterable of python dicts

    .. note::

        The default implementation enforces ``type_safe`` True
    """
    return as_fugue_df(df).as_dict_iterable(columns=columns)