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
def _does_run_match_clause(cls, run, sed):
    key_type = sed.get('type')
    key = sed.get('key')
    value = sed.get('value')
    comparator = sed.get('comparator').upper()
    key = SearchUtils.translate_key_alias(key)
    if cls.is_metric(key_type, comparator):
        lhs = run.data.metrics.get(key, None)
        value = float(value)
    elif cls.is_param(key_type, comparator):
        lhs = run.data.params.get(key, None)
    elif cls.is_tag(key_type, comparator):
        lhs = run.data.tags.get(key, None)
    elif cls.is_string_attribute(key_type, key, comparator):
        lhs = getattr(run.info, key)
    elif cls.is_numeric_attribute(key_type, key, comparator):
        lhs = getattr(run.info, key)
        value = int(value)
    elif cls.is_dataset(key_type, comparator):
        if key == 'context':
            return any((SearchUtils.get_comparison_func(comparator)(tag.value if tag else None, value) for dataset_input in run.inputs.dataset_inputs for tag in dataset_input.tags if tag.key == MLFLOW_DATASET_CONTEXT))
        else:
            return any((SearchUtils.get_comparison_func(comparator)(getattr(dataset_input.dataset, key), value) for dataset_input in run.inputs.dataset_inputs))
    else:
        raise MlflowException(f"Invalid search expression type '{key_type}'", error_code=INVALID_PARAMETER_VALUE)
    if lhs is None:
        return False
    return SearchUtils.get_comparison_func(comparator)(lhs, value)