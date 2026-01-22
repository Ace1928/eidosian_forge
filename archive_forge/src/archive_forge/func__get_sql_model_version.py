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
@classmethod
def _get_sql_model_version(cls, session, name, version, eager=False):
    """
        Args:
            eager: If ``True``, eagerly loads the model version's tags.
                If ``False``, these attributes are not eagerly loaded and
                will be loaded when their corresponding object properties
                are accessed from the resulting ``SqlModelVersion`` object.
        """
    _validate_model_name(name)
    _validate_model_version(version)
    query_options = cls._get_eager_model_version_query_options() if eager else []
    conditions = [SqlModelVersion.name == name, SqlModelVersion.version == version, SqlModelVersion.current_stage != STAGE_DELETED_INTERNAL]
    return cls._get_model_version_from_db(session, name, version, conditions, query_options)