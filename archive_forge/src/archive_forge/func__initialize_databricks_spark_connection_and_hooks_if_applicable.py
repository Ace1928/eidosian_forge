import abc
import json
import logging
import os
import time
import traceback
from enum import Enum
from typing import Any, Dict, List, Optional
import yaml
from mlflow.recipes.cards import CARD_HTML_NAME, CARD_PICKLE_NAME, BaseCard, FailureCard
from mlflow.recipes.utils import get_recipe_name
from mlflow.recipes.utils.step import display_html
from mlflow.tracking import MlflowClient
from mlflow.utils.databricks_utils import is_in_databricks_runtime
def _initialize_databricks_spark_connection_and_hooks_if_applicable(self) -> None:
    """
        Initializes a connection to the Databricks Spark Gateway and sets up associated hooks
        (e.g. MLflow Run creation notification hooks) if MLflow Recipes is running in the
        Databricks Runtime.
        """
    if is_in_databricks_runtime():
        try:
            from dbruntime.spark_connection import initialize_spark_connection, is_pinn_mode_enabled
            from IPython.utils.io import capture_output
            with capture_output():
                spark_handles, entry_point = initialize_spark_connection(is_pinn_mode_enabled())
        except Exception as e:
            _logger.warning('Encountered unexpected failure while initializing Spark connection. Spark operations may not succeed. Exception: %s', e)
        else:
            try:
                from dbruntime.MlflowCreateRunHook import get_mlflow_create_run_hook
                get_mlflow_create_run_hook(spark_handles['sc'], entry_point)
            except Exception as e:
                _logger.warning('Encountered unexpected failure while setting up Databricks MLflow Run creation hooks. Exception: %s', e)