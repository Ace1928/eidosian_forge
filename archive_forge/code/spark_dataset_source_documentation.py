from typing import Any, Dict, Optional
from mlflow.data.dataset_source import DatasetSource
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
Loads the dataset source as a Spark Dataset Source.

        Returns:
            An instance of ``pyspark.sql.DataFrame``.

        