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
def _get_registered_model_tag(cls, session, name, key):
    tags = session.query(SqlRegisteredModelTag).filter(SqlRegisteredModelTag.name == name, SqlRegisteredModelTag.key == key).all()
    if len(tags) == 0:
        return None
    if len(tags) > 1:
        raise MlflowException(f'Expected only 1 registered model tag with name={name}, key={key}. Found {len(tags)}.', INVALID_STATE)
    return tags[0]