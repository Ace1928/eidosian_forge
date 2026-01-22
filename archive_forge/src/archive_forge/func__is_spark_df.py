import logging
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.types import DataType
from mlflow.types.schema import (
def _is_spark_df(x) -> bool:
    try:
        import pyspark.sql.dataframe
        if isinstance(x, pyspark.sql.dataframe.DataFrame):
            return True
    except ImportError:
        return False
    try:
        import pyspark.sql.connect.dataframe
        return isinstance(x, pyspark.sql.connect.dataframe.DataFrame)
    except ImportError:
        return False