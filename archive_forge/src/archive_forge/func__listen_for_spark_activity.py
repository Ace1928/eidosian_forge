import concurrent.futures
import logging
import sys
import threading
import uuid
from py4j.java_gateway import CallbackServerParameters
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.spark import FLAVOR_NAME
from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.utils import _truncate_and_ellipsize
from mlflow.utils.autologging_utils import (
from mlflow.utils.databricks_utils import get_repl_id as get_databricks_repl_id
from mlflow.utils.validation import MAX_TAG_VAL_LENGTH
def _listen_for_spark_activity(spark_context):
    global _spark_table_info_listener
    if _get_current_listener() is not None:
        return
    if _get_spark_major_version(spark_context) < 3:
        raise MlflowException('Spark autologging unsupported for Spark versions < 3')
    gw = spark_context._gateway
    params = gw.callback_server_parameters
    callback_server_params = CallbackServerParameters(address=params.address, port=params.port, daemonize=True, daemonize_connections=True, eager_load=params.eager_load, ssl_context=params.ssl_context, accept_timeout=params.accept_timeout, read_timeout=params.read_timeout, auth_token=params.auth_token)
    callback_server_started = gw.start_callback_server(callback_server_params)
    try:
        event_publisher = _get_jvm_event_publisher(spark_context)
        event_publisher.init(1)
        _spark_table_info_listener = PythonSubscriber()
        event_publisher.register(_spark_table_info_listener)
    except Exception as e:
        if callback_server_started:
            try:
                gw.shutdown_callback_server()
            except Exception as e:
                _logger.warning('Failed to shut down Spark callback server for autologging: %s', str(e))
        _spark_table_info_listener = None
        raise MlflowException('Exception while attempting to initialize JVM-side state for Spark datasource autologging. Note that Spark datasource autologging only works with Spark 3.0 and above. Please create a new Spark session with required Spark version and ensure you have the mlflow-spark JAR attached to your Spark session as described in https://mlflow.org/docs/latest/tracking/autolog.html#spark Exception:\n%s' % e)
    from mlflow.tracking.context.registry import _run_context_provider_registry
    _run_context_provider_registry.register(SparkAutologgingContext)
    _logger.info('Autologging successfully enabled for spark.')