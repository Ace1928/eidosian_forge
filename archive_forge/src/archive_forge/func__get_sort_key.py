import ast
import base64
import json
import math
import operator
import re
import shlex
import sqlparse
from packaging.version import Version
from sqlparse.sql import (
from sqlparse.tokens import Token as TokenType
from mlflow.entities import RunInfo
from mlflow.entities.model_registry.model_version_stages import STAGE_DELETED_INTERNAL
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.db.db_types import MSSQL, MYSQL, POSTGRES, SQLITE
from mlflow.utils.mlflow_tags import (
@classmethod
def _get_sort_key(cls, order_by_list):
    order_by = []
    parsed_order_by = map(cls.parse_order_by_for_search_model_versions, order_by_list or [])
    for type_, key, ascending in parsed_order_by:
        if type_ == 'attribute':
            if key == 'version_number':
                key = 'version'
            order_by.append((key, ascending))
        else:
            raise MlflowException.invalid_parameter_value(f'Invalid order_by entity: {type_}')
    if not any((key == 'name' for key, _ in order_by)):
        order_by.append(('name', True))
    if not any((key == 'version_number' for key, _ in order_by)):
        order_by.append(('version', False))
    return lambda model_version: tuple((_apply_reversor(model_version, k, asc) for k, asc in order_by))