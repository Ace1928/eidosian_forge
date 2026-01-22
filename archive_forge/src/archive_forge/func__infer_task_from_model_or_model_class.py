import importlib
import inspect
import itertools
import os
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import huggingface_hub
from packaging import version
from requests.exceptions import ConnectionError as RequestsConnectionError
from transformers import AutoConfig, PretrainedConfig, is_tf_available, is_torch_available
from transformers.utils import SAFE_WEIGHTS_NAME, TF2_WEIGHTS_NAME, WEIGHTS_NAME, logging
from ..utils import CONFIG_NAME
from ..utils.import_utils import is_onnx_available
@classmethod
def _infer_task_from_model_or_model_class(cls, model: Optional[Union['PreTrainedModel', 'TFPreTrainedModel']]=None, model_class: Optional[Type]=None) -> str:
    if model is not None and model_class is not None:
        raise ValueError('Either a model or a model class must be provided, but both were given here.')
    if model is None and model_class is None:
        raise ValueError('Either a model or a model class must be provided, but none were given here.')
    target_name = model.__class__.__name__ if model is not None else model_class.__name__
    task_name = None
    iterable = ()
    for _, model_loader in cls._LIBRARY_TO_MODEL_LOADERS_TO_TASKS_MAP.items():
        iterable += (model_loader.items(),)
    for _, model_loader in cls._LIBRARY_TO_TF_MODEL_LOADERS_TO_TASKS_MAP.items():
        iterable += (model_loader.items(),)
    pt_auto_module = importlib.import_module('transformers.models.auto.modeling_auto')
    tf_auto_module = importlib.import_module('transformers.models.auto.modeling_tf_auto')
    for auto_cls_name, task in itertools.chain.from_iterable(iterable):
        if any((target_name.startswith('Auto'), target_name.startswith('TFAuto'), 'StableDiffusion' in target_name)):
            if target_name == auto_cls_name:
                task_name = task
                break
            continue
        module = tf_auto_module if auto_cls_name.startswith('TF') else pt_auto_module
        auto_cls = getattr(module, auto_cls_name, None)
        if auto_cls is None:
            continue
        model_mapping = auto_cls._model_mapping._model_mapping
        if target_name in model_mapping.values():
            task_name = task
            break
    if task_name is None:
        raise ValueError(f'Could not infer the task name for {target_name}.')
    return task_name