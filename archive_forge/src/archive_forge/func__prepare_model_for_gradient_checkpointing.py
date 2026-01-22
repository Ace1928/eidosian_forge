from __future__ import annotations
import os
from contextlib import contextmanager
from typing import Any, Optional, Union
import torch
from accelerate.hooks import remove_hook_from_submodules
from torch import nn
from transformers.utils import PushToHubMixin
from peft.tuners.mixed import COMPATIBLE_TUNER_TYPES
from .config import PeftConfig
from .peft_model import PeftModel
from .tuners import (
from .utils import PeftType, _set_adapter, _set_trainable
def _prepare_model_for_gradient_checkpointing(model: nn.Module) -> None:
    """
    Prepares the model for gradient checkpointing if necessary
    """
    if not getattr(model, 'is_gradient_checkpointing', True):
        return model
    if not (getattr(model, 'is_loaded_in_8bit', False) or getattr(model, 'is_loaded_in_4bit', False) or getattr(model, 'is_quantized', False)):
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        elif hasattr(model, 'get_input_embeddings'):

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)