from typing import List, Optional
from sqlalchemy.exc import IntegrityError, MultipleResultsFound, NoResultFound
from sqlalchemy.orm import sessionmaker
from werkzeug.security import check_password_hash, generate_password_hash
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
from mlflow.server.auth.db import utils as dbutils
from mlflow.server.auth.db.models import (
from mlflow.server.auth.entities import ExperimentPermission, RegisteredModelPermission, User
from mlflow.server.auth.permissions import _validate_permission
from mlflow.store.db.utils import _get_managed_session_maker, create_sqlalchemy_engine_with_retry
from mlflow.utils.uri import extract_db_type_from_uri
from mlflow.utils.validation import _validate_username
def _get_registered_model_permission(self, session, name: str, username: str) -> SqlRegisteredModelPermission:
    try:
        user = self._get_user(session, username=username)
        return session.query(SqlRegisteredModelPermission).filter(SqlRegisteredModelPermission.name == name, SqlRegisteredModelPermission.user_id == user.id).one()
    except NoResultFound:
        raise MlflowException(f'Registered model permission with name={name} and username={username} not found', RESOURCE_DOES_NOT_EXIST)
    except MultipleResultsFound:
        raise MlflowException(f'Found multiple registered model permissions with name={name} and username={username}', INVALID_STATE)