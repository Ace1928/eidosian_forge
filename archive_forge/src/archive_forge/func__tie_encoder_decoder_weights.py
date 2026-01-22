import collections
import copy
import functools
import gc
import importlib.metadata
import inspect
import itertools
import json
import os
import re
import shutil
import tempfile
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from zipfile import is_zipfile
import torch
from packaging import version
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss, Identity
from torch.utils.checkpoint import checkpoint
from .activations import get_activation
from .configuration_utils import PretrainedConfig
from .dynamic_module_utils import custom_object_save
from .generation import GenerationConfig, GenerationMixin
from .integrations import PeftAdapterMixin, deepspeed_config, is_deepspeed_zero3_enabled
from .pytorch_utils import (  # noqa: F401
from .quantizers import AutoHfQuantizer, HfQuantizer
from .safetensors_conversion import auto_conversion
from .utils import (
from .utils.hub import convert_file_size_to_int, create_and_tag_model_card, get_checkpoint_shard_files
from .utils.import_utils import (
from .utils.quantization_config import BitsAndBytesConfig, QuantizationMethod
@staticmethod
def _tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module, base_model_prefix: str):
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        logger.info(f'{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized.')

    def tie_encoder_to_decoder_recursively(decoder_pointer: nn.Module, encoder_pointer: nn.Module, module_name: str, uninitialized_encoder_weights: List[str], depth=0):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(encoder_pointer, nn.Module), f'{decoder_pointer} and {encoder_pointer} have to be of type nn.Module'
        if hasattr(decoder_pointer, 'weight'):
            assert hasattr(encoder_pointer, 'weight')
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, 'bias'):
                assert hasattr(encoder_pointer, 'bias')
                encoder_pointer.bias = decoder_pointer.bias
            return
        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert len(encoder_modules) > 0, f'Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}'
            all_encoder_weights = {module_name + '/' + sub_name for sub_name in encoder_modules.keys()}
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(encoder_modules) != len(decoder_modules):
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError('Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model.')
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(decoder_modules[decoder_name], encoder_modules[encoder_name], module_name + '/' + name, uninitialized_encoder_weights, depth=depth + 1)
                all_encoder_weights.remove(module_name + '/' + encoder_name)
            uninitialized_encoder_weights += list(all_encoder_weights)
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights)
    if len(uninitialized_encoder_weights) > 0:
        logger.warning(f'The following encoder weights were not tied to the decoder {uninitialized_encoder_weights}')