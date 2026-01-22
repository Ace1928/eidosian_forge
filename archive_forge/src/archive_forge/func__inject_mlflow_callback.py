import contextlib
import inspect
import logging
import uuid
import warnings
from copy import deepcopy
from packaging.version import Version
import mlflow
from mlflow.entities import RunTag
from mlflow.exceptions import MlflowException
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.tracking.context import registry as context_registry
from mlflow.utils.autologging_utils import (
from mlflow.utils.autologging_utils.safety import _resolve_extra_tags
def _inject_mlflow_callback(func_name, mlflow_callback, args, kwargs):
    if func_name == 'invoke':
        from langchain.schema.runnable.config import RunnableConfig
        in_args = False
        if len(args) >= 2:
            config = args[1]
            in_args = True
        else:
            config = kwargs.get('config', None)
        if config is None:
            callbacks = [mlflow_callback]
            config = RunnableConfig(callbacks=callbacks)
        else:
            callbacks = config.get('callbacks') or []
            if not isinstance(callbacks, list):
                callbacks = [callbacks]
            config['callbacks'] = [*callbacks, mlflow_callback]
        if in_args:
            args = (args[0], config) + args[2:]
        else:
            kwargs['config'] = config
        return (args, kwargs)
    if func_name == '__call__':
        if len(args) >= 3:
            callbacks = args[2] or []
            if not isinstance(callbacks, list):
                callbacks = [callbacks]
            args = args[:2] + ([*callbacks, mlflow_callback],) + args[3:]
        else:
            callbacks = kwargs.get('callbacks') or []
            if not isinstance(callbacks, list):
                callbacks = [callbacks]
            kwargs['callbacks'] = [*callbacks, mlflow_callback]
        return (args, kwargs)
    if func_name == 'get_relevant_documents':
        callbacks = kwargs.get('callbacks') or []
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        kwargs['callbacks'] = [*callbacks, mlflow_callback]
        return (args, kwargs)