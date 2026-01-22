import copy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from packaging import version
from transformers.models.speecht5.modeling_speecht5 import SpeechT5HifiGan
from transformers.utils import is_tf_available, is_torch_available
from ...utils import (
from ...utils.import_utils import _diffusers_version
from ..tasks import TasksManager
from .constants import ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME, ONNX_ENCODER_NAME
def _get_submodels_and_onnx_configs(model: Union['PreTrainedModel', 'TFPreTrainedModel'], task: str, monolith: bool, custom_onnx_configs: Dict, custom_architecture: bool, _variant: str, library_name: str, int_dtype: str='int64', float_dtype: str='fp32', fn_get_submodels: Optional[Callable]=None, preprocessors: Optional[List[Any]]=None, legacy: bool=False, model_kwargs: Optional[Dict]=None):
    if not custom_architecture:
        if library_name == 'diffusers':
            onnx_config = None
            models_and_onnx_configs = get_stable_diffusion_models_for_export(model, int_dtype=int_dtype, float_dtype=float_dtype)
        else:
            onnx_config_constructor = TasksManager.get_exporter_config_constructor(model=model, exporter='onnx', task=task, library_name=library_name)
            onnx_config = onnx_config_constructor(model.config, int_dtype=int_dtype, float_dtype=float_dtype, preprocessors=preprocessors, legacy=legacy)
            onnx_config.variant = _variant
            all_variants = '\n'.join([f'    - {name}: {description}' for name, description in onnx_config.VARIANTS.items()])
            logger.info(f'Using the export variant {onnx_config.variant}. Available variants are:\n{all_variants}')
            if model.config.is_encoder_decoder and task.startswith(TasksManager._ENCODER_DECODER_TASKS) and (not monolith):
                models_and_onnx_configs = get_encoder_decoder_models_for_export(model, onnx_config)
            elif task.startswith('text-generation') and (not monolith):
                models_and_onnx_configs = get_decoder_models_for_export(model, onnx_config, legacy=legacy)
            elif model.config.model_type == 'sam':
                models_and_onnx_configs = get_sam_models_for_export(model, onnx_config)
            elif model.config.model_type == 'speecht5':
                models_and_onnx_configs = get_speecht5_models_for_export(model, onnx_config, model_kwargs)
            else:
                models_and_onnx_configs = {'model': (model, onnx_config)}
        for key, custom_onnx_config in custom_onnx_configs.items():
            models_and_onnx_configs[key] = (models_and_onnx_configs[key][0], custom_onnx_config)
    else:
        onnx_config = None
        submodels_for_export = None
        models_and_onnx_configs = {}
        if fn_get_submodels is not None:
            submodels_for_export = fn_get_submodels(model)
        elif library_name == 'diffusers':
            submodels_for_export = _get_submodels_for_export_stable_diffusion(model)
        elif model.config.is_encoder_decoder and task.startswith(TasksManager._ENCODER_DECODER_TASKS) and (not monolith):
            submodels_for_export = _get_submodels_for_export_encoder_decoder(model, use_past=task.endswith('-with-past'))
        elif task.startswith('text-generation') and (not monolith):
            submodels_for_export = _get_submodels_for_export_decoder(model, use_past=task.endswith('-with-past'))
        else:
            submodels_for_export = {'model': model}
        if submodels_for_export.keys() != custom_onnx_configs.keys():
            logger.error(f'ONNX custom configs for: {', '.join(custom_onnx_configs.keys())}')
            logger.error(f'Submodels to export: {', '.join(submodels_for_export.keys())}')
            raise ValueError('Trying to export a custom model, but could not find as many custom ONNX configs as the number of submodels to export. Please specifiy the fn_get_submodels argument, that should return a dictionary of submodules with as many items as the provided custom_onnx_configs dictionary.')
        for key, custom_onnx_config in custom_onnx_configs.items():
            models_and_onnx_configs[key] = (submodels_for_export[key], custom_onnx_config)
    if onnx_config is None:
        onnx_config = next(iter(models_and_onnx_configs.values()))[1]
    return (onnx_config, models_and_onnx_configs)