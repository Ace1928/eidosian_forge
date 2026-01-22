import os
import pathlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import pandas as pd
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from triad.collections.dict import ParamDict
from triad.collections.schema import Schema
from triad.utils.assertion import assert_or_throw
from triad.utils.io import join, url_to_fs
from triad.utils.pandas_like import PD_UTILS
from fugue.dataframe import LocalBoundedDataFrame, LocalDataFrame, PandasDataFrame
def _safe_load_csv(p: FileParser, **kwargs: Any) -> pd.DataFrame:

    def load_dir() -> pd.DataFrame:
        dfs: List[pd.DataFrame] = []
        for _p in p.join('*.csv').find_all():
            with _p.open('r') as f:
                dfs.append(pd.read_csv(f, **kwargs))
        return pd.concat(dfs)
    try:
        with p.open('r') as f:
            return pd.read_csv(f, **kwargs)
    except IsADirectoryError:
        return load_dir()
    except pd.errors.ParserError:
        return load_dir()
    except PermissionError:
        return load_dir()