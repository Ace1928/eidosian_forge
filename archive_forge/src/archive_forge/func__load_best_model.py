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
def _load_best_model(self):
    logger.info(f'Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).')
    best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
    best_safe_model_path = os.path.join(self.state.best_model_checkpoint, SAFE_WEIGHTS_NAME)
    best_adapter_model_path = os.path.join(self.state.best_model_checkpoint, ADAPTER_WEIGHTS_NAME)
    best_safe_adapter_model_path = os.path.join(self.state.best_model_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)
    model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
    if self.is_deepspeed_enabled:
        deepspeed_load_checkpoint(self.model_wrapped, self.state.best_model_checkpoint, load_module_strict=not _is_peft_model(self.model))
    elif self.is_fsdp_enabled:
        load_result = load_fsdp_model(self.accelerator.state.fsdp_plugin, self.accelerator, model, self.state.best_model_checkpoint, **_get_fsdp_ckpt_kwargs())
    elif os.path.exists(best_model_path) or os.path.exists(best_safe_model_path) or os.path.exists(best_adapter_model_path) or os.path.exists(best_safe_adapter_model_path):
        has_been_loaded = True
        weights_only_kwarg = {'weights_only': True} if is_torch_greater_or_equal_than_1_13 else {}
        if is_sagemaker_mp_enabled():
            if os.path.isfile(os.path.join(self.state.best_model_checkpoint, 'user_content.pt')):
                smp.resume_from_checkpoint(path=self.state.best_model_checkpoint, tag=WEIGHTS_NAME, partial=False, load_optimizer=False)
            else:
                if self.args.save_safetensors and os.path.isfile(best_safe_model_path):
                    state_dict = safetensors.torch.load_file(best_safe_model_path, device='cpu')
                else:
                    state_dict = torch.load(best_model_path, map_location='cpu', **weights_only_kwarg)
                state_dict['_smp_is_partial'] = False
                load_result = model.load_state_dict(state_dict, strict=True)
        else:
            if _is_peft_model(model):
                if hasattr(model, 'active_adapter') and hasattr(model, 'load_adapter'):
                    if os.path.exists(best_adapter_model_path) or os.path.exists(best_safe_adapter_model_path):
                        model.load_adapter(self.state.best_model_checkpoint, model.active_adapter)
                        from torch.nn.modules.module import _IncompatibleKeys
                        load_result = _IncompatibleKeys([], [])
                    else:
                        logger.warning(f'The intermediate checkpoints of PEFT may not be saved correctly, consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. Check some examples here: https://github.com/huggingface/peft/issues/96')
                        has_been_loaded = False
                else:
                    logger.warning('Could not load adapter model, make sure to have `peft>=0.3.0` installed')
                    has_been_loaded = False
            else:
                if self.args.save_safetensors and os.path.isfile(best_safe_model_path):
                    state_dict = safetensors.torch.load_file(best_safe_model_path, device='cpu')
                else:
                    state_dict = torch.load(best_model_path, map_location='cpu', **weights_only_kwarg)
                load_result = model.load_state_dict(state_dict, False)
            if not is_sagemaker_mp_enabled() and has_been_loaded:
                self._issue_warnings_after_load(load_result)
    elif os.path.exists(os.path.join(self.state.best_model_checkpoint, WEIGHTS_INDEX_NAME)):
        load_result = load_sharded_checkpoint(model, self.state.best_model_checkpoint, strict=is_sagemaker_mp_enabled())
        if not is_sagemaker_mp_enabled():
            self._issue_warnings_after_load(load_result)
    else:
        logger.warning(f'Could not locate the best model at {best_model_path}, if you are running a distributed training on multiple nodes, you should activate `--save_on_each_node`.')