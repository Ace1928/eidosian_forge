from __future__ import annotations
import math
import operator
import re
import warnings
from dataclasses import asdict, replace
from enum import Enum
from functools import reduce
from itertools import chain
from typing import Literal, Optional
import torch
from torch import nn
from tqdm import tqdm
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists, onload_layer
from peft.utils import (
from peft.utils.merge_utils import dare_linear, dare_ties, magnitude_prune, task_arithmetic, ties
from .aqlm import dispatch_aqlm
from .awq import dispatch_awq
from .config import LoraConfig
from .gptq import dispatch_gptq
from .layer import Conv2d, LoraLayer, dispatch_default
from .tp_layer import dispatch_megatron
def _svd_generalized_task_arithmetic_weighted_adapter(self, combination_type, adapters, weights, new_rank, target, target_lora_A, target_lora_B, density, majority_sign_method, clamp=None, full_matrices=True, driver=None):
    valid_adapters = []
    valid_weights = []
    is_embedding = any((adapter in target.lora_embedding_A for adapter in adapters))
    for adapter, weight in zip(adapters, weights):
        if adapter in target.lora_A or adapter in target.lora_embedding_A:
            valid_adapters.append(adapter)
            valid_weights.append(weight * target.scaling[adapter])
    if len(valid_adapters) == 0:
        raise ValueError('No matching LoRAs found. Please raise an issue on Github.')
    delta_weight = [target.get_delta_weight(adapter) for adapter in valid_adapters]
    valid_weights = torch.tensor(valid_weights).to(delta_weight[0].device)
    if combination_type == 'svd':
        delta_weight = task_arithmetic(delta_weight, valid_weights)
    elif combination_type == 'ties_svd':
        delta_weight = ties(delta_weight, valid_weights, density, majority_sign_method)
    elif combination_type == 'dare_linear_svd':
        delta_weight = dare_linear(delta_weight, valid_weights, density)
    elif combination_type == 'dare_ties_svd':
        delta_weight = dare_ties(delta_weight, valid_weights, density, majority_sign_method)
    elif combination_type == 'magnitude_prune_svd':
        delta_weight = magnitude_prune(delta_weight, valid_weights, density)
    else:
        raise ValueError(f'Invalid value passed to combination type: {combination_type}')
    conv2d = isinstance(target, Conv2d)
    if conv2d:
        conv2d_1x1 = target.weight.size()[2:4] == (1, 1)
        if not conv2d_1x1:
            delta_weight = delta_weight.flatten(start_dim=1)
        else:
            delta_weight = delta_weight.squeeze()
    if hasattr(target, 'fan_in_fan_out') and target.fan_in_fan_out or is_embedding:
        delta_weight = delta_weight.T
    U, S, Vh = torch.linalg.svd(delta_weight, full_matrices=full_matrices, driver=driver)
    U = U[:, :new_rank]
    S = S[:new_rank]
    U = U @ torch.diag(S)
    Vh = Vh[:new_rank, :]
    if clamp is not None:
        dist = torch.cat([U.flatten(), Vh.flatten()])
        hi_val = torch.quantile(dist, clamp)
        low_val = -hi_val
        U = U.clamp(low_val, hi_val)
        Vh = Vh.clamp(low_val, hi_val)
    if conv2d:
        U = U.reshape(target_lora_B.data.shape)
        Vh = Vh.reshape(target_lora_A.data.shape)
    return (Vh, U)