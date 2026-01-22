import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from .integrations import (
import huggingface_hub.utils as hf_hub_utils
import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import ModelCard, create_repo, upload_folder
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from . import __version__
from .configuration_utils import PretrainedConfig
from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from .debug_utils import DebugOption, DebugUnderflowOverflow
from .hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
from .integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from .integrations.tpu import tpu_spmd_dataloader
from .modelcard import TrainingSummary
from .modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from .models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from .optimization import Adafactor, get_scheduler
from .pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from .tokenization_utils_base import PreTrainedTokenizerBase
from .trainer_callback import (
from .trainer_pt_utils import (
from .trainer_utils import (
from .training_args import OptimizerNames, ParallelMode, TrainingArguments
from .utils import (
from .utils.quantization_config import QuantizationMethod
def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
    if model is None:
        model = self.model
    config_file = os.path.join(resume_from_checkpoint, CONFIG_NAME)
    adapter_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_WEIGHTS_NAME)
    adapter_safe_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)
    weights_file = os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
    weights_index_file = os.path.join(resume_from_checkpoint, WEIGHTS_INDEX_NAME)
    safe_weights_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_NAME)
    safe_weights_index_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_INDEX_NAME)
    is_fsdp_ckpt = os.path.isdir(resume_from_checkpoint) and (any((FSDP_MODEL_NAME in folder_name for folder_name in os.listdir(resume_from_checkpoint) if os.path.isdir(os.path.join(resume_from_checkpoint, folder_name)))) or os.path.isfile(os.path.join(resume_from_checkpoint, f'{FSDP_MODEL_NAME}.bin')))
    if is_fsdp_ckpt and (not self.is_fsdp_enabled):
        raise ValueError(f'Checkpoint found at {resume_from_checkpoint} is only supported when using PyTorch FSDP')
    if not (any((os.path.isfile(f) for f in [weights_file, safe_weights_file, weights_index_file, safe_weights_index_file, adapter_weights_file, adapter_safe_weights_file])) or is_fsdp_ckpt):
        raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")
    logger.info(f'Loading model from {resume_from_checkpoint}.')
    if os.path.isfile(config_file):
        config = PretrainedConfig.from_json_file(config_file)
        checkpoint_version = config.transformers_version
        if checkpoint_version is not None and checkpoint_version != __version__:
            logger.warning(f'You are resuming training from a checkpoint trained with {checkpoint_version} of Transformers but your current version is {__version__}. This is not recommended and could yield to errors or unwanted behaviors.')
    if os.path.isfile(weights_file) or os.path.isfile(safe_weights_file) or is_fsdp_ckpt:
        weights_only_kwarg = {'weights_only': True} if is_torch_greater_or_equal_than_1_13 else {}
        if is_sagemaker_mp_enabled():
            if os.path.isfile(os.path.join(resume_from_checkpoint, 'user_content.pt')):
                smp.resume_from_checkpoint(path=resume_from_checkpoint, tag=WEIGHTS_NAME, partial=False, load_optimizer=False)
            else:
                if hasattr(self.args, 'fp16') and self.args.fp16 is True:
                    logger.warning('Enabling FP16 and loading from smp < 1.10 checkpoint together is not suppported.')
                state_dict = torch.load(weights_file, map_location='cpu', **weights_only_kwarg)
                state_dict['_smp_is_partial'] = False
                load_result = model.load_state_dict(state_dict, strict=True)
                del state_dict
        elif self.is_fsdp_enabled:
            load_fsdp_model(self.accelerator.state.fsdp_plugin, self.accelerator, model, resume_from_checkpoint, **_get_fsdp_ckpt_kwargs())
        else:
            if self.args.save_safetensors and os.path.isfile(safe_weights_file):
                state_dict = safetensors.torch.load_file(safe_weights_file, device='cpu')
            else:
                state_dict = torch.load(weights_file, map_location='cpu', **weights_only_kwarg)
            load_result = model.load_state_dict(state_dict, False)
            del state_dict
            self._issue_warnings_after_load(load_result)
    elif _is_peft_model(model):
        if hasattr(model, 'active_adapter') and hasattr(model, 'load_adapter'):
            if os.path.exists(resume_from_checkpoint):
                model.load_adapter(resume_from_checkpoint, model.active_adapter, is_trainable=True)
            else:
                logger.warning(f'The intermediate checkpoints of PEFT may not be saved correctly, consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. Check some examples here: https://github.com/huggingface/peft/issues/96')
        else:
            logger.warning('Could not load adapter model, make sure to have `peft>=0.3.0` installed')
    else:
        load_result = load_sharded_checkpoint(model, resume_from_checkpoint, strict=is_sagemaker_mp_enabled(), prefer_safe=self.args.save_safetensors)
        if not is_sagemaker_mp_enabled():
            self._issue_warnings_after_load(load_result)