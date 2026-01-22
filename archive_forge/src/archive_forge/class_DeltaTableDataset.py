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
class DeltaTableDataset(_SparkDatasetMixin, _LocationBasedDataset):
    """
    Representation of a dataset in delta format with files having the `.delta` extension.
    """

    def __init__(self, location: str, dataset_format: str, recipe_root: str, version: Optional[int]=None, timestamp: Optional[str]=None):
        """
        Args:
            location: The location of the dataset (e.g. '/tmp/myfile.parquet', './mypath',
                's3://mybucket/mypath', ...).
            dataset_format: The format of the dataset (e.g. 'csv', 'parquet', ...).
            recipe_root: The absolute path of the associated recipe root directory on the
                local filesystem.
            version: The version of the Delta table to read.
            timestamp: The timestamp at which to read the Delta table.
        """
        super().__init__(location=location, dataset_format=dataset_format, recipe_root=recipe_root)
        self.version = version
        self.timestamp = timestamp

    def resolve_to_parquet(self, dst_path: str):
        spark_session = self._get_or_create_spark_session()
        spark_read_op = spark_session.read.format('delta')
        if self.version is not None:
            spark_read_op = spark_read_op.option('versionAsOf', self.version)
        if self.timestamp is not None:
            spark_read_op = spark_read_op.option('timestampAsOf', self.timestamp)
        spark_df = spark_read_op.load(self.location)
        pandas_df = self._convert_spark_df_to_pandas(spark_df)
        write_pandas_df_as_parquet(df=pandas_df, data_parquet_path=dst_path)

    @staticmethod
    def handles_format(dataset_format: str) -> bool:
        return dataset_format == 'delta'

    @classmethod
    def _from_config(cls, dataset_config: Dict[str, Any], recipe_root: str) -> '_Dataset':
        return cls(location=cls._get_required_config(dataset_config=dataset_config, key='location'), recipe_root=recipe_root, dataset_format=cls._get_required_config(dataset_config=dataset_config, key='using'), version=dataset_config.get('version'), timestamp=dataset_config.get('timestamp'))