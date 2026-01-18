import os
import pathlib
import posixpath
import re
import urllib.parse
import uuid
from typing import Any, Tuple
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.utils.os import is_windows
from mlflow.utils.validation import _validate_db_type_string
def validate_query_string(query):
    query = _decode(query)
    if '..' in query:
        raise MlflowException('Invalid query string', error_code=INVALID_PARAMETER_VALUE)