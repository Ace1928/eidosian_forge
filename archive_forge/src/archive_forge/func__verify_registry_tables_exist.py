import logging
import urllib
import sqlalchemy
from sqlalchemy.future import select
import mlflow.store.db.utils
from mlflow.entities.model_registry.model_version_stages import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
from mlflow.store.artifact.utils.models import _parse_model_uri
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry import (
from mlflow.store.model_registry.abstract_store import AbstractStore
from mlflow.store.model_registry.dbmodels.models import (
from mlflow.utils.search_utils import SearchModelUtils, SearchModelVersionUtils, SearchUtils
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.uri import extract_db_type_from_uri
from mlflow.utils.validation import (
@staticmethod
def _verify_registry_tables_exist(engine):
    inspected_tables = set(sqlalchemy.inspect(engine).get_table_names())
    expected_tables = [SqlRegisteredModel.__tablename__, SqlModelVersion.__tablename__]
    if any((table not in inspected_tables for table in expected_tables)):
        raise MlflowException('Database migration in unexpected state. Run manual upgrade.')