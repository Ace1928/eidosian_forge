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
def _get_input_data_from_function(func_name, model, args, kwargs):
    func_param_name_mapping = {'__call__': 'inputs', 'invoke': 'input', 'get_relevant_documents': 'query'}
    input_example_exc = None
    if (param_name := func_param_name_mapping.get(func_name)):
        inference_func = getattr(model, func_name)
        if next(iter(inspect.signature(inference_func).parameters.keys())) != param_name:
            input_example_exc = MlflowException('Inference function signature changes, please contact MLflow team to fix langchain autologging.')
        else:
            return args[0] if len(args) > 0 else kwargs.get(param_name)
    else:
        input_example_exc = MlflowException(f'Unsupported inference function. Only support {list(func_param_name_mapping.keys())}.')
    _logger.warning(f'Failed to gather input example of model {model.__class__.__name__} due to {input_example_exc}.')