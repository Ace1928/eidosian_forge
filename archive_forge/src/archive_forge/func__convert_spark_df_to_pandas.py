import importlib
import logging
import os
import pathlib
import posixpath
import sys
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INVALID_PARAMETER_VALUE
from mlflow.store.artifact.artifact_repo import (
from mlflow.utils._spark_utils import (
from mlflow.utils.file_utils import (
def _convert_spark_df_to_pandas(self, spark_df):
    import pandas as pd
    datetime_cols = [field.name for field in spark_df.schema.fields if str(field.dataType) == 'DateType']
    pandas_df = spark_df.toPandas()
    pandas_df[datetime_cols] = pandas_df[datetime_cols].apply(pd.to_datetime, errors='coerce')
    return pandas_df