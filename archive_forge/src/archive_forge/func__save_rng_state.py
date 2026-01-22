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
def _save_rng_state(self, output_dir):
    rng_states = {'python': random.getstate(), 'numpy': np.random.get_state(), 'cpu': torch.random.get_rng_state()}
    if torch.cuda.is_available():
        if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
            rng_states['cuda'] = torch.cuda.random.get_rng_state_all()
        else:
            rng_states['cuda'] = torch.cuda.random.get_rng_state()
    if is_torch_tpu_available():
        rng_states['xla'] = xm.get_rng_state()
    if is_torch_npu_available():
        if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
            rng_states['npu'] = torch.npu.random.get_rng_state_all()
        else:
            rng_states['npu'] = torch.npu.random.get_rng_state()
    os.makedirs(output_dir, exist_ok=True)
    if self.args.world_size <= 1:
        torch.save(rng_states, os.path.join(output_dir, 'rng_state.pth'))
    else:
        torch.save(rng_states, os.path.join(output_dir, f'rng_state_{self.args.process_index}.pth'))