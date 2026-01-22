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
def _search_runs(self, experiment_ids, filter_string, run_view_type, max_results, order_by, page_token):

    def compute_next_token(current_size):
        next_token = None
        if max_results == current_size:
            final_offset = offset + max_results
            next_token = SearchUtils.create_page_token(final_offset)
        return next_token
    if max_results > SEARCH_MAX_RESULTS_THRESHOLD:
        raise MlflowException(f'Invalid value for request parameter max_results. It must be at most {SEARCH_MAX_RESULTS_THRESHOLD}, but got value {max_results}', INVALID_PARAMETER_VALUE)
    stages = set(LifecycleStage.view_type_to_stages(run_view_type))
    with self.ManagedSessionMaker() as session:
        parsed_filters = SearchUtils.parse_search_filter(filter_string)
        cases_orderby, parsed_orderby, sorting_joins = _get_orderby_clauses(order_by, session)
        stmt = select(SqlRun, *cases_orderby)
        attribute_filters, non_attribute_filters, dataset_filters = _get_sqlalchemy_filter_clauses(parsed_filters, session, self._get_dialect())
        for non_attr_filter in non_attribute_filters:
            stmt = stmt.join(non_attr_filter)
        for idx, dataset_filter in enumerate(dataset_filters):
            anon_table_name = f'anon_{idx + 1}'
            stmt = stmt.join(dataset_filter, text(f'runs.run_uuid = {anon_table_name}.destination_id'))
        for j in sorting_joins:
            stmt = stmt.outerjoin(j)
        offset = SearchUtils.parse_start_offset_from_page_token(page_token)
        stmt = stmt.distinct().options(*self._get_eager_run_query_options()).filter(SqlRun.experiment_id.in_(experiment_ids), SqlRun.lifecycle_stage.in_(stages), *attribute_filters).order_by(*parsed_orderby).offset(offset).limit(max_results)
        queried_runs = session.execute(stmt).scalars(SqlRun).all()
        runs = [run.to_mlflow_entity() for run in queried_runs]
        run_ids = [run.info.run_id for run in runs]
        inputs = self._get_run_inputs(run_uuids=run_ids, session=session)
        runs_with_inputs = []
        for i, run in enumerate(runs):
            runs_with_inputs.append(Run(run.info, run.data, RunInputs(dataset_inputs=inputs[i])))
        next_page_token = compute_next_token(len(runs_with_inputs))
    return (runs_with_inputs, next_page_token)