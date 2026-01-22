import os
from pathlib import Path
from typing import Union
import cloudpickle
import yaml
from mlflow.exceptions import MlflowException
from mlflow.langchain.utils import (
def _save_internal_runnables(runnable, path, loader_fn, persist_dir):
    conf = {}
    if isinstance(runnable, lc_runnables_types()):
        conf[_MODEL_TYPE_KEY] = runnable.__class__.__name__
        conf.update(_save_runnables(runnable, path, loader_fn, persist_dir))
    elif isinstance(runnable, base_lc_types()):
        lc_model = _validate_and_wrap_lc_model(runnable, loader_fn)
        conf[_MODEL_TYPE_KEY] = lc_model.__class__.__name__
        conf.update(_save_base_lcs(lc_model, path, loader_fn, persist_dir))
    else:
        conf = {_MODEL_TYPE_KEY: runnable.__class__.__name__, _MODEL_DATA_KEY: _MODEL_DATA_YAML_FILE_NAME, _MODEL_LOAD_KEY: _CONFIG_LOAD_KEY}
        path = path / _MODEL_DATA_YAML_FILE_NAME
        if hasattr(runnable, 'save'):
            runnable.save(path)
        elif hasattr(runnable, 'dict'):
            runnable_dict = runnable.dict()
            with open(path, 'w') as f:
                yaml.dump(runnable_dict, f, default_flow_style=False)
        else:
            return Exception(f'Runnable {runnable} is not supported for saving.')
    return conf