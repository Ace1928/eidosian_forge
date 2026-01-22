import argparse
import copy
import enum
import functools
import os
import typing
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, get_args
import torch
from .constants import FSDP_AUTO_WRAP_POLICY, FSDP_BACKWARD_PREFETCH, FSDP_SHARDING_STRATEGY, FSDP_STATE_DICT_TYPE
from .environment import str_to_bool
from .imports import is_cuda_available, is_npu_available, is_xpu_available
from .versions import compare_versions
def _deepspeed_config_checks(self):
    env_variable_names_to_ignore = ['ACCELERATE_GRADIENT_ACCUMULATION_STEPS', 'ACCELERATE_GRADIENT_CLIPPING', 'ACCELERATE_DEEPSPEED_ZERO_STAGE', 'ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE', 'ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_DEVICE', 'ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_NVME_PATH', 'ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_NVME_PATH', 'ACCELERATE_DEEPSPEED_ZERO3_SAVE_16BIT_MODEL', 'ACCELERATE_MIXED_PRECISION']
    env_variable_names_to_ignore = [name.replace('ACCELERATE_', '').replace('DEEPSPEED_', '').lower() for name in env_variable_names_to_ignore]
    deepspeed_fields_from_accelerate_config = os.environ.get('ACCELERATE_CONFIG_DS_FIELDS', '').split(',')
    if any((name in env_variable_names_to_ignore for name in deepspeed_fields_from_accelerate_config)):
        raise ValueError(f'When using `deepspeed_config_file`, the following accelerate config variables will be ignored: {env_variable_names_to_ignore}.\nPlease specify them appropriately in the DeepSpeed config file.\nIf you are using an accelerate config file, remove others config variables mentioned in the above specified list.\nThe easiest method is to create a new config following the questionnaire via `accelerate config`.\nIt will only ask for the necessary config variables when using `deepspeed_config_file`.')