import os
from pathlib import Path
from typing import Union
import cloudpickle
import yaml
from mlflow.exceptions import MlflowException
from mlflow.langchain.utils import (
def _save_runnables(model, path, loader_fn=None, persist_dir=None):
    model_data_kwargs = {_MODEL_LOAD_KEY: _RUNNABLE_LOAD_KEY}
    if isinstance(model, lc_runnable_with_steps_types()):
        model_data_path = _MODEL_DATA_FOLDER_NAME
        _save_runnable_with_steps(model, os.path.join(path, model_data_path), loader_fn, persist_dir)
    elif isinstance(model, picklable_runnable_types()):
        model_data_path = _MODEL_DATA_PKL_FILE_NAME
        _save_picklable_runnable(model, os.path.join(path, model_data_path))
    elif isinstance(model, lc_runnable_branch_types()):
        model_data_path = _MODEL_DATA_FOLDER_NAME
        _save_runnable_branch(model, os.path.join(path, model_data_path), loader_fn, persist_dir)
    elif isinstance(model, lc_runnable_assign_types()):
        model_data_path = _MODEL_DATA_FOLDER_NAME
        _save_runnable_assign(model, os.path.join(path, model_data_path), loader_fn, persist_dir)
    else:
        raise MlflowException.invalid_parameter_value(_UNSUPPORTED_MODEL_ERROR_MESSAGE.format(instance_type=type(model).__name__))
    model_data_kwargs.update({_MODEL_DATA_KEY: model_data_path})
    return model_data_kwargs