import logging
import os
import types
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Optional, Union
import torch
from packaging.version import parse
from ..utils import check_if_pytorch_greater, is_accelerate_available, recurse_getattr, recurse_setattr
from .models import BetterTransformerManager
class BetterTransformer(object):
    """
    A conversion wrapper that takes as an input the `transformers` model to be converted
    and returns the converted `BetterTransformer` model. The `BetterTransformer` model is based on the `BetterTransformer`
    recently released by PyTorch from its 1.12 version:
    https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/

    # Original PR from: https://github.com/huggingface/transformers/pull/19553 adapted and wrapped in this script.
    """

    @check_if_pytorch_greater('1.13.99', 'Please upgrade PyTorch following https://pytorch.org/get-started/locally/ in order to use BetterTransformer.')
    def transform(model: torch.nn.Module, keep_original_model: bool=False, max_memory: Optional[Dict]=None, offload_dir: Optional[Union[str, os.PathLike]]=None, **kwargs) -> torch.nn.Module:
        """
        Conversion script from `transformers` model to its BetterTransformers version

        Args:
            model (`torch.nn.Module`):
                Original `transformers` model
            keep_original_model (`bool`, defaults to `False`):
                whether to keep or override the original model - essentially
                for memory efficiency reasons
            max_memory (`Optional[Dict]`, defaults to `None`):
                Same argument as `max_memory` argument from `.from_pretrained` function
                in `transformers`.
        Returns:
            The converted model if the conversion has been successful.
        """
        hf_config = model.config
        if hf_config.model_type in ['falcon', 'gpt_bigcode', 'llama', 'whisper']:
            raise ValueError(f'Transformers now supports natively BetterTransformer optimizations (torch.nn.functional.scaled_dot_product_attention) for the model type {hf_config.model_type}. As such, there is no need to use `model.to_bettertransformers()` or `BetterTransformer.transform(model)` from the Optimum library. Please upgrade to transformers>=4.36 and torch>=2.1.1 to use it. Details: https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-and-memory-efficient-attention-through-pytorchs-scaleddotproductattention.')
        if hasattr(model, 'hf_device_map'):
            load_accelerate = True
            hf_device_map = model.hf_device_map
        else:
            load_accelerate = False
        if hasattr(model, 'use_bettertransformer') and model.use_bettertransformer is True:
            raise Exception('`BetterTransform.transform()` was called on a model already using Better Transformer modeling.')
        if BetterTransformerManager.cannot_support(model.config.model_type):
            raise ValueError(f'The model type {model.config.model_type} can not be supported to be used with BetterTransformer. The identified reason is: {BetterTransformerManager.CAN_NOT_BE_SUPPORTED[model.config.model_type]}. Currently supported models are: {BetterTransformerManager.MODEL_MAPPING.keys()}.')
        if not BetterTransformerManager.supports(model.config.model_type):
            raise NotImplementedError(f'The model type {model.config.model_type} is not yet supported to be used with BetterTransformer. Feel free to open an issue at https://github.com/huggingface/optimum/issues if you would like this model type to be supported. Currently supported models are: {BetterTransformerManager.MODEL_MAPPING.keys()}.')
        if parse(torch.__version__) <= parse('1.14'):
            raise ValueError(f'BetterTransformer requires torch>=2.0 but {torch.__version__} is installed. Please upgrade PyTorch.')
        if load_accelerate:
            remove_hook_from_module(model, recurse=True)
        training_mode = model.training
        if keep_original_model:
            try:
                if not check_if_pytorch_greater(2.0, 'Please upgrade PyTorch to >=2.0 to use training mode'):
                    model = model.requires_grad_(False)
                model_fast = deepcopy(model)
            except RuntimeError:
                raise ValueError(f'The model {model.__class__.__name__} does not support `deepcopy` operation that is internally used to create a copy of the original model when using `keep_original_model=True`. Please run the conversion with `keep_original_model=False` and create a new copy of the original model somewhere else.')
            model_fast = replace_to_bettertransformer(model_fast, hf_config)
        else:
            model_fast = replace_to_bettertransformer(model, hf_config)
            model = None
        if BetterTransformerManager.requires_nested_tensor(model_fast.config.model_type):
            set_last_layer(model_fast)
        setattr(model_fast, 'use_bettertransformer', True)
        if load_accelerate:
            all_model_tensors = [name for name, _ in model_fast.state_dict().items()]
            for module_name in hf_device_map.keys():
                all_model_tensors = [name for name in all_model_tensors if not name.startswith(module_name)]
            if len(all_model_tensors) > 0:
                bt_device_map = infer_auto_device_map(model_fast, max_memory=max_memory)
            else:
                bt_device_map = hf_device_map
            model_fast = dispatch_model(model_fast, bt_device_map, offload_dir=offload_dir)
            if keep_original_model:
                model = dispatch_model(model, hf_device_map, offload_dir=offload_dir)
        logger.warning('The BetterTransformer implementation does not support padding during training, as the fused kernels do not support attention masks. Beware that passing padded batched data during training may result in unexpected outputs. Please refer to https://huggingface.co/docs/optimum/bettertransformer/overview for more details.')
        model_fast._old_save_pretrained = model_fast.save_pretrained
        model_fast._old_push_to_hub = model_fast.push_to_hub
        model_fast.save_pretrained = raise_save_or_push_incompatible
        model_fast.push_to_hub = raise_save_or_push_incompatible
        if training_mode:
            model_fast = model_fast.train()
        else:
            model_fast = model_fast.eval()
        return model_fast

    def reverse(bt_model: 'PreTrainedModel') -> 'PreTrainedModel':
        """
        Converts back a model using BetterTransformer to its canonical transformers modeling implementation, in order to save
        and share it.

        Args:
            bt_model (`PreTrainedModel`):
                Model using BetterTransform to convert back to use transformers modeling.

        Returns:
            PreTrainedModel: _description_
        """
        if getattr(bt_model, 'use_bettertransformer', False) is False:
            raise ValueError('The method BetterTransformer.reverse() should be used on a model already transformed to the BetterTransformer format, which appears to not be the case.')
        if parse(torch.__version__) <= parse('1.14'):
            raise ValueError(f'BetterTransformer reverse transform requires torch>=2.0 but {torch.__version__} is installed. Please upgrade PyTorch.')
        config = bt_model.config
        if config.model_type not in ['wav2vec2', 'hubert', 'bark']:
            with torch.device('meta'):
                reversed_model = bt_model.__class__(config)
        else:
            logger.warning('The reverse transform for the architectures wav2vec2, hubert, bark is memory-heavy due to a bug in PyTorch.')
            reversed_model = bt_model.__class__(config)
        if bt_model.training is False:
            reversed_model = reversed_model.eval()
        reversed_modules_paths = []
        for path, module in reversed_model.named_modules():
            if path.startswith(tuple(reversed_modules_paths)):
                continue
            if config.model_type in BetterTransformerManager.EXCLUDE_FROM_TRANSFORM and any((subname in path for subname in BetterTransformerManager.EXCLUDE_FROM_TRANSFORM[config.model_type])):
                continue
            target_classes = list(BetterTransformerManager.MODEL_MAPPING[config.model_type].keys())
            has_been_replaced = False
            for target_class in target_classes:
                if module.__class__.__name__ == target_class:
                    has_been_replaced = True
                    break
            if has_been_replaced:
                recurse_setattr(reversed_model, path, recurse_getattr(bt_model, path)._revert(module))
                reversed_modules_paths.append(path + '.')
        for path, param in reversed_model.state_dict().items():
            if param.device == torch.device('meta') or not path.startswith(tuple(reversed_modules_paths)):
                recurse_setattr(reversed_model, path, recurse_getattr(bt_model, path))
        for path, param in reversed_model.named_buffers():
            if param.device == torch.device('meta') or not path.startswith(tuple(reversed_modules_paths)):
                recurse_setattr(reversed_model, path, recurse_getattr(bt_model, path))
        return reversed_model