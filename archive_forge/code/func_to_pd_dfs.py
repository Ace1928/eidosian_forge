from typing import Any, Dict, List, Tuple
from unittest import TestCase
import numpy as np
import pandas as pd
from pandasql import sqldf
from datetime import datetime, timedelta
from qpd.dataframe import DataFrame, Column
from qpd import run_sql
from qpd.qpd_engine import QPDEngine
from qpd_test.utils import assert_df_eq
def to_pd_dfs(self, dfs: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    res: Dict[str, DataFrame] = {}
    for k, v in dfs.items():
        if isinstance(v, tuple):
            res[k] = pd.DataFrame(v[0], columns=v[1])
        elif isinstance(v, pd.DataFrame):
            res[k] = v
        else:
            raise NotImplementedError
    return res