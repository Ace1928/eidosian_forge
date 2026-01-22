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
def _save_optimizer_and_scheduler(self, output_dir):
    if is_torch_tpu_available():
        xm.rendezvous('saving_optimizer_states')
        xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
        with warnings.catch_warnings(record=True) as caught_warnings:
            xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)
    elif is_sagemaker_mp_enabled():
        opt_state_dict = self.optimizer.local_state_dict(gather_if_shard=False)
        smp.barrier()
        if smp.rdp_rank() == 0 or smp.state.cfg.shard_optimizer_state:
            smp.save(opt_state_dict, os.path.join(output_dir, OPTIMIZER_NAME), partial=True, v3=smp.state.cfg.shard_optimizer_state)
    elif self.is_deepspeed_enabled:
        accept_exclude_frozen_parameters = 'exclude_frozen_parameters' in set(inspect.signature(self.model_wrapped.save_checkpoint).parameters.keys())
        if accept_exclude_frozen_parameters and _is_peft_model(self.model):
            self.model_wrapped.save_checkpoint(output_dir, exclude_frozen_parameters=True)
        else:
            self.model_wrapped.save_checkpoint(output_dir)
    elif self.is_fsdp_enabled:
        save_fsdp_model(self.accelerator.state.fsdp_plugin, self.accelerator, self.model, output_dir, **_get_fsdp_ckpt_kwargs())
        save_fsdp_optimizer(self.accelerator.state.fsdp_plugin, self.accelerator, self.optimizer, self.model, output_dir)
    elif self.args.should_save:
        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
    is_deepspeed_custom_scheduler = self.is_deepspeed_enabled and (not isinstance(self.lr_scheduler, DeepSpeedSchedulerWrapper))
    if self.args.should_save and (not self.is_deepspeed_enabled or is_deepspeed_custom_scheduler) and (not is_torch_tpu_available()):
        with warnings.catch_warnings(record=True) as caught_warnings:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
        reissue_pt_warnings(caught_warnings)