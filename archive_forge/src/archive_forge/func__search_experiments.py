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
def _search_experiments(self, view_type, max_results, filter_string, order_by, page_token):

    def compute_next_token(current_size):
        next_token = None
        if max_results + 1 == current_size:
            final_offset = offset + max_results
            next_token = SearchExperimentsUtils.create_page_token(final_offset)
        return next_token
    if not isinstance(max_results, int) or max_results < 1:
        raise MlflowException(f'Invalid value for max_results. It must be a positive integer, but got {max_results}', INVALID_PARAMETER_VALUE)
    if max_results > SEARCH_MAX_RESULTS_THRESHOLD:
        raise MlflowException(f'Invalid value for max_results. It must be at most {SEARCH_MAX_RESULTS_THRESHOLD}, but got {max_results}', INVALID_PARAMETER_VALUE)
    with self.ManagedSessionMaker() as session:
        parsed_filters = SearchExperimentsUtils.parse_search_filter(filter_string)
        attribute_filters, non_attribute_filters = _get_search_experiments_filter_clauses(parsed_filters, self._get_dialect())
        order_by_clauses = _get_search_experiments_order_by_clauses(order_by)
        offset = SearchUtils.parse_start_offset_from_page_token(page_token)
        lifecycle_stags = set(LifecycleStage.view_type_to_stages(view_type))
        stmt = reduce(lambda s, f: s.join(f), non_attribute_filters, select(SqlExperiment)).options(*self._get_eager_experiment_query_options()).filter(*attribute_filters, SqlExperiment.lifecycle_stage.in_(lifecycle_stags)).order_by(*order_by_clauses).offset(offset).limit(max_results + 1)
        queried_experiments = session.execute(stmt).scalars(SqlExperiment).all()
        experiments = [e.to_mlflow_entity() for e in queried_experiments]
        next_page_token = compute_next_token(len(experiments))
    return (experiments[:max_results], next_page_token)