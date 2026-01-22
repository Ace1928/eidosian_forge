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
class SparkSqlDataset(_SparkDatasetMixin, _Dataset):
    """
    Representation of a Spark SQL dataset defined by a Spark SQL query string
    (e.g. `SELECT * FROM my_spark_table`).
    """

    def __init__(self, sql: str, location: str, dataset_format: str):
        """
        Args:
            sql: The Spark SQL query string that defines the dataset
                (e.g. 'SELECT * FROM my_spark_table').
            location: The location of the dataset
                (e.g. 'catalog.schema.table', 'schema.table', 'table').
            dataset_format: The format of the dataset (e.g. 'csv', 'parquet', ...).
        """
        super().__init__(dataset_format=dataset_format)
        self.sql = sql
        self.location = location

    def resolve_to_parquet(self, dst_path: str):
        if self.location is None and self.sql is None:
            raise MlflowException('Either location or sql configuration key must be specified for dataset with format spark_sql') from None
        spark_session = self._get_or_create_spark_session()
        spark_df = None
        if self.sql is not None:
            spark_df = spark_session.sql(self.sql)
        elif self.location is not None:
            spark_df = spark_session.table(self.location)
        pandas_df = self._convert_spark_df_to_pandas(spark_df)
        write_pandas_df_as_parquet(df=pandas_df, data_parquet_path=dst_path)

    @classmethod
    def _from_config(cls, dataset_config: Dict[str, Any], recipe_root: str) -> '_Dataset':
        return cls(sql=dataset_config.get('sql'), location=dataset_config.get('location'), dataset_format=cls._get_required_config(dataset_config=dataset_config, key='using'))

    @staticmethod
    def handles_format(dataset_format: str) -> bool:
        return dataset_format == 'spark_sql'