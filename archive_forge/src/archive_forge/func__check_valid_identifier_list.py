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
def _check_valid_identifier_list(cls, value_token):
    if len(value_token._groupable_tokens) == 0:
        raise MlflowException('While parsing a list in the query, expected a non-empty list of string values, but got empty list', error_code=INVALID_PARAMETER_VALUE)
    if len(value_token._groupable_tokens) == 1 and value_token._groupable_tokens[0].ttype is TokenType.String.Single:
        return
    if not isinstance(value_token._groupable_tokens[0], IdentifierList):
        raise MlflowException('While parsing a list in the query, expected a non-empty list of string values, but got ill-formed list.', error_code=INVALID_PARAMETER_VALUE)
    elif not all((token.ttype in {*cls.STRING_VALUE_TYPES, *cls.DELIMITER_VALUE_TYPES, cls.WHITESPACE_VALUE_TYPE} for token in value_token._groupable_tokens[0].tokens)):
        raise MlflowException(f'While parsing a list in the query, expected string value, punctuation, or whitespace, but got different type in list: {value_token}', error_code=INVALID_PARAMETER_VALUE)