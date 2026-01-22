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
class DeltaDatasetSource(DatasetSource):
    """
    Represents the source of a dataset stored at in a delta table.
    """

    def __init__(self, path: Optional[str]=None, delta_table_name: Optional[str]=None, delta_table_version: Optional[int]=None, delta_table_id: Optional[str]=None):
        if (path, delta_table_name).count(None) != 1:
            raise MlflowException('Must specify exactly one of "path" or "table_name"', INVALID_PARAMETER_VALUE)
        self._path = path
        if delta_table_name is not None:
            self._delta_table_name = get_full_name_from_sc(delta_table_name, _get_active_spark_session())
        else:
            self._delta_table_name = delta_table_name
        self._delta_table_version = delta_table_version
        self._delta_table_id = delta_table_id

    @staticmethod
    def _get_source_type() -> str:
        return 'delta_table'

    def load(self, **kwargs):
        """
        Loads the dataset source as a Delta Dataset Source.

        Returns:
            An instance of ``pyspark.sql.DataFrame``.
        """
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        spark_read_op = spark.read.format('delta')
        if self._delta_table_version is not None:
            spark_read_op = spark_read_op.option('versionAsOf', self._delta_table_version)
        if self._path:
            return spark_read_op.load(self._path)
        else:
            return spark_read_op.table(self._delta_table_name)

    @property
    def path(self) -> Optional[str]:
        return self._path

    @property
    def delta_table_name(self) -> Optional[str]:
        return self._delta_table_name

    @property
    def delta_table_id(self) -> Optional[str]:
        return self._delta_table_id

    @property
    def delta_table_version(self) -> Optional[int]:
        return self._delta_table_version

    @staticmethod
    def _can_resolve(raw_source: Any):
        return False

    @classmethod
    def _resolve(cls, raw_source: str) -> 'DeltaDatasetSource':
        raise NotImplementedError

    def _is_databricks_uc_table(self):
        if self._delta_table_name is not None:
            catalog_name = self._delta_table_name.split('.', 1)[0]
            return catalog_name not in DATABRICKS_LOCAL_METASTORE_NAMES and catalog_name != DATABRICKS_SAMPLES_CATALOG_NAME
        else:
            return False

    def _lookup_table_id(self, table_name):
        try:
            req_body = message_to_json(GetTable(full_name_arg=table_name))
            _METHOD_TO_INFO = extract_api_info_for_service(UnityCatalogService, _REST_API_PATH_PREFIX)
            db_creds = get_databricks_host_creds()
            endpoint, method = _METHOD_TO_INFO[GetTable]
            final_endpoint = endpoint.replace('{full_name_arg}', table_name)
            resp = call_endpoint(host_creds=db_creds, endpoint=final_endpoint, method=method, json_body=req_body, response_proto=GetTableResponse)
            return resp.table_id
        except Exception:
            return None

    def to_dict(self) -> Dict[Any, Any]:
        info = {}
        if self._path:
            info['path'] = self._path
        if self._delta_table_name:
            info['delta_table_name'] = self._delta_table_name
        if self._delta_table_version:
            info['delta_table_version'] = self._delta_table_version
        if self._is_databricks_uc_table():
            info['is_databricks_uc_table'] = True
            if self._delta_table_id:
                info['delta_table_id'] = self._delta_table_id
            else:
                info['delta_table_id'] = self._lookup_table_id(self._delta_table_name)
        return info

    @classmethod
    def from_dict(cls, source_dict: Dict[Any, Any]) -> 'DeltaDatasetSource':
        return cls(path=source_dict.get('path'), delta_table_name=source_dict.get('delta_table_name'), delta_table_version=source_dict.get('delta_table_version'), delta_table_id=source_dict.get('delta_table_id'))