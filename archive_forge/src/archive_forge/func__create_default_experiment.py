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
def _create_default_experiment(self, session):
    """
        MLflow UI and client code expects a default experiment with ID 0.
        This method uses SQL insert statement to create the default experiment as a hack, since
        experiment table uses 'experiment_id' column is a PK and is also set to auto increment.
        MySQL and other implementation do not allow value '0' for such cases.

        ToDo: Identify a less hacky mechanism to create default experiment 0
        """
    table = SqlExperiment.__tablename__
    creation_time = get_current_time_millis()
    default_experiment = {SqlExperiment.experiment_id.name: int(SqlAlchemyStore.DEFAULT_EXPERIMENT_ID), SqlExperiment.name.name: Experiment.DEFAULT_EXPERIMENT_NAME, SqlExperiment.artifact_location.name: str(self._get_artifact_location(0)), SqlExperiment.lifecycle_stage.name: LifecycleStage.ACTIVE, SqlExperiment.creation_time.name: creation_time, SqlExperiment.last_update_time.name: creation_time}

    def decorate(s):
        if is_string_type(s):
            return repr(s)
        else:
            return str(s)
    columns = list(default_experiment.keys())
    values = ', '.join([decorate(default_experiment.get(c)) for c in columns])
    try:
        self._set_zero_value_insertion_for_autoincrement_column(session)
        session.execute(sql.text(f'INSERT INTO {table} ({', '.join(columns)}) VALUES ({values});'))
    finally:
        self._unset_zero_value_insertion_for_autoincrement_column(session)