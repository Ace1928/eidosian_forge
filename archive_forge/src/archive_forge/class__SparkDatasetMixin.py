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
class _SparkDatasetMixin:
    """
    Mixin class providing Spark-related utilities for Datasets that use Spark for resolution
    and conversion to parquet format.
    """

    def _convert_spark_df_to_pandas(self, spark_df):
        import pandas as pd
        datetime_cols = [field.name for field in spark_df.schema.fields if str(field.dataType) == 'DateType']
        pandas_df = spark_df.toPandas()
        pandas_df[datetime_cols] = pandas_df[datetime_cols].apply(pd.to_datetime, errors='coerce')
        return pandas_df

    def _get_or_create_spark_session(self):
        """
        Obtains the active Spark session, throwing if a session does not exist.

        Returns:
            The active Spark session.
        """
        try:
            spark_session = _get_active_spark_session()
            if spark_session:
                _logger.debug('Found active spark session')
            else:
                spark_session = _create_local_spark_session_for_recipes()
                _logger.debug('Creating new spark session')
            return spark_session
        except Exception as e:
            raise MlflowException(message=f"Encountered an error while searching for an active Spark session to load the dataset with format '{self.dataset_format}'. Please create a Spark session and try again.", error_code=BAD_REQUEST) from e