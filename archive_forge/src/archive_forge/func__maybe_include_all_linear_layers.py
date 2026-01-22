from __future__ import annotations
import logging
import re
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Optional, Union
import torch
from accelerate.hooks import AlignDevicesHook
from accelerate.utils import named_module_tensors, offload_state_dict
from torch import nn
from transformers import PreTrainedModel
from transformers.pytorch_utils import Conv1D
from peft.utils import INCLUDE_LINEAR_LAYERS_SHORTHAND
from ..config import PeftConfig
from ..utils import ModulesToSaveWrapper, _get_submodules
def _maybe_include_all_linear_layers(peft_config: PeftConfig, model: nn.Module) -> PeftConfig:
    """
    Helper function to update `target_modules` to all linear/Conv1D layers if provided as 'all-linear'. Adapted from
    the QLoRA repository: https://github.com/artidoro/qlora/blob/main/qlora.py
    """
    if not (isinstance(peft_config.target_modules, str) and peft_config.target_modules.lower() == INCLUDE_LINEAR_LAYERS_SHORTHAND):
        return peft_config
    if not isinstance(model, PreTrainedModel):
        raise ValueError(f'Only instances of PreTrainedModel support `target_modules={INCLUDE_LINEAR_LAYERS_SHORTHAND!r}`')
    linear_classes = (torch.nn.Linear, Conv1D)
    linear_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_classes):
            names = name.rsplit('.', 1)[-1]
            linear_module_names.add(names)
    output_emb = model.get_output_embeddings()
    if output_emb is not None:
        last_module_name = [name for name, module in model.named_modules() if module is output_emb][0]
        linear_module_names -= {last_module_name}
    peft_config.target_modules = linear_module_names
    return peft_config