import logging
from typing import Any, Dict, Optional
from mlflow.data.dataset_source import DatasetSource
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_managed_catalog_messages_pb2 import (
from mlflow.protos.databricks_managed_catalog_service_pb2 import UnityCatalogService
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils._unity_catalog_utils import get_full_name_from_sc
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import (
def _is_databricks_uc_table(self):
    if self._delta_table_name is not None:
        catalog_name = self._delta_table_name.split('.', 1)[0]
        return catalog_name not in DATABRICKS_LOCAL_METASTORE_NAMES and catalog_name != DATABRICKS_SAMPLES_CATALOG_NAME
    else:
        return False