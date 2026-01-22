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
def _get_comparison(cls, comparison):
    stripped_comparison = [token for token in comparison.tokens if not token.is_whitespace]
    cls._validate_comparison(stripped_comparison)
    left, comparator, right = stripped_comparison
    comp = cls._get_model_version_search_identifier(left.value, cls.VALID_SEARCH_ATTRIBUTE_KEYS)
    comp['comparator'] = comparator.value.upper()
    comp['value'] = cls._get_value(comp.get('type'), comp.get('key'), right)
    return comp