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
def _generalized_task_arithmetic_weighted_adapter(self, combination_type, adapters, weights, target, density, majority_sign_method):
    valid_weights = []
    lora_A_deltas = []
    lora_B_deltas = []
    for adapter, weight in zip(adapters, weights):
        if adapter in target.lora_A:
            current_adapter_lora_A = target.lora_A[adapter].weight
            current_adapter_lora_B = target.lora_B[adapter].weight
        elif adapter in target.lora_embedding_A:
            current_adapter_lora_A = target.lora_embedding_A[adapter]
            current_adapter_lora_B = target.lora_embedding_B[adapter]
        else:
            continue
        valid_weights.append(math.sqrt(weight * target.scaling[adapter]))
        lora_A_deltas.append(current_adapter_lora_A.data)
        lora_B_deltas.append(current_adapter_lora_B.data)
    valid_weights = torch.tensor(valid_weights).to(lora_A_deltas[0].device)
    lora_deltas = [lora_A_deltas, lora_B_deltas]
    dtype = lora_A_deltas[0].dtype
    for i, task_tensors in enumerate(lora_deltas):
        if combination_type == 'linear':
            lora_deltas[i] = task_arithmetic(task_tensors, valid_weights)
        elif combination_type == 'ties':
            lora_deltas[i] = ties(task_tensors, valid_weights, density, majority_sign_method)
        elif combination_type == 'dare_linear':
            lora_deltas[i] = dare_linear(task_tensors, valid_weights, density)
        elif combination_type == 'dare_ties':
            lora_deltas[i] = dare_ties(task_tensors, valid_weights, density, majority_sign_method)
        elif combination_type == 'magnitude_prune':
            lora_deltas[i] = magnitude_prune(task_tensors, valid_weights, density)
        else:
            raise ValueError('Invalid combination type')
    lora_deltas = [delta.to(dtype) for delta in lora_deltas]
    return lora_deltas