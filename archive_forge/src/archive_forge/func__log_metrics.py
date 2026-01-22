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
def _log_metrics(self, run_id, metrics):
    if not metrics:
        return
    metric_instances = []
    seen = set()
    for metric in metrics:
        metric, value, is_nan = self._get_metric_value_details(metric)
        if metric not in seen:
            metric_instances.append(SqlMetric(run_uuid=run_id, key=metric.key, value=value, timestamp=metric.timestamp, step=metric.step, is_nan=is_nan))
        seen.add(metric)
    with self.ManagedSessionMaker() as session:
        run = self._get_run(run_uuid=run_id, session=session)
        self._check_run_is_active(run)

        def _insert_metrics(metric_instances):
            session.add_all(metric_instances)
            self._update_latest_metrics_if_necessary(metric_instances, session)
            session.commit()
        try:
            _insert_metrics(metric_instances)
        except sqlalchemy.exc.IntegrityError:
            session.rollback()
            metric_keys = [m.key for m in metric_instances]
            metric_key_batches = [metric_keys[i:i + 100] for i in range(0, len(metric_keys), 100)]
            for metric_key_batch in metric_key_batches:
                metric_history = session.query(SqlMetric).filter(SqlMetric.run_uuid == run_id, SqlMetric.key.in_(metric_key_batch)).all()
                metric_history = {m.to_mlflow_entity() for m in metric_history}
                non_existing_metrics = [m for m in metric_instances if m.to_mlflow_entity() not in metric_history]
                _insert_metrics(non_existing_metrics)