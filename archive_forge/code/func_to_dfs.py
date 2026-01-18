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
def to_dfs(self, dfs: Dict[str, Any]) -> Dict[str, DataFrame]:
    res: Dict[str, DataFrame] = {}
    for k, v in dfs.items():
        if isinstance(v, DataFrame):
            res[k] = v
        elif isinstance(v, tuple):
            res[k] = self.to_df(v[0], v[1])
        else:
            res[k] = self.to_df(v, None)
    return res