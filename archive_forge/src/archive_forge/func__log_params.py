import json
import logging
import math
import random
import threading
import time
import uuid
from functools import reduce
from typing import List, Optional
import sqlalchemy
import sqlalchemy.sql.expression as sql
from sqlalchemy import and_, func, sql, text
from sqlalchemy.future import select
import mlflow.store.db.utils
from mlflow.entities import (
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.entities.metric import MetricWithRunId
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
from mlflow.store.db.db_types import MSSQL, MYSQL
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT, SEARCH_MAX_RESULTS_THRESHOLD
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.store.tracking.dbmodels.models import (
from mlflow.utils.file_utils import local_file_uri_to_path, mkdir
from mlflow.utils.mlflow_tags import (
from mlflow.utils.name_utils import _generate_random_name
from mlflow.utils.search_utils import SearchExperimentsUtils, SearchUtils
from mlflow.utils.string_utils import is_string_type
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.uri import (
from mlflow.utils.validation import (
def _log_params(self, run_id, params):
    if not params:
        return
    with self.ManagedSessionMaker() as session:
        run = self._get_run(run_uuid=run_id, session=session)
        self._check_run_is_active(run)
        existing_params = {p.key: p.value for p in run.params}
        new_params = []
        non_matching_params = []
        for param in params:
            if param.key in existing_params:
                if param.value != existing_params[param.key]:
                    non_matching_params.append({'key': param.key, 'old_value': existing_params[param.key], 'new_value': param.value})
                continue
            new_params.append(SqlParam(run_uuid=run_id, key=param.key, value=param.value))
        if non_matching_params:
            raise MlflowException(f"Changing param values is not allowed. Params were already logged='{non_matching_params}' for run ID='{run_id}'.", INVALID_PARAMETER_VALUE)
        if not new_params:
            return
        session.add_all(new_params)