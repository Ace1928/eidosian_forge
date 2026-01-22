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
def _parse_order_by_string(cls, order_by):
    token_value = cls._validate_order_by_and_generate_token(order_by)
    is_ascending = True
    tokens = shlex.split(token_value.replace('`', '"'))
    if len(tokens) > 2:
        raise MlflowException(f"Invalid order_by clause '{order_by}'. Could not be parsed.", error_code=INVALID_PARAMETER_VALUE)
    elif len(tokens) == 2:
        order_token = tokens[1].lower()
        if order_token not in cls.VALID_ORDER_BY_TAGS:
            raise MlflowException(f"Invalid ordering key in order_by clause '{order_by}'.", error_code=INVALID_PARAMETER_VALUE)
        is_ascending = order_token == cls.ASC_OPERATOR
        token_value = tokens[0]
    return (token_value, is_ascending)