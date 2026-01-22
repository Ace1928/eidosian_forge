import os
from pathlib import Path
from typing import Union
import cloudpickle
import yaml
from mlflow.exceptions import MlflowException
from mlflow.langchain.utils import (
def _load_model_from_config(path, model_config):
    from langchain.chains.loading import type_to_loader_dict as chains_type_to_loader_dict
    from langchain.llms import get_type_to_cls_dict as llms_get_type_to_cls_dict
    try:
        from langchain.prompts.loading import type_to_loader_dict as prompts_types
    except ImportError:
        prompts_types = {'prompt', 'few_shot_prompt'}
    config_path = os.path.join(path, model_config.get(_MODEL_DATA_KEY, _MODEL_DATA_YAML_FILE_NAME))
    if config_path.endswith('.yaml'):
        config = _load_from_yaml(config_path)
    elif config_path.endswith('.json'):
        config = _load_from_json(config_path)
    else:
        raise MlflowException(f'Cannot load runnable without a config file. Got path {config_path}.')
    _type = config.get('_type')
    if _type in chains_type_to_loader_dict:
        from langchain.chains.loading import load_chain
        return _patch_loader(load_chain)(config_path)
    elif _type in prompts_types:
        from langchain.prompts.loading import load_prompt
        return load_prompt(config_path)
    elif _type in llms_get_type_to_cls_dict():
        from langchain.llms.loading import load_llm
        return _patch_loader(load_llm)(config_path)
    elif _type in custom_type_to_loader_dict():
        return custom_type_to_loader_dict()[_type](config)
    raise MlflowException(f'Unsupported type {_type} for loading.')