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
def to_native_df(self, data: Any, columns: Any) -> Any:
    raise NotImplementedError