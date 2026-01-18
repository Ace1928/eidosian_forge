import importlib
import warnings
from typing import Any, Optional
import torch
import torch.nn as nn
import torch.nn.init as init
from peft.tuners.tuners_utils import BaseTunerLayer
from .layer import LoraLayer
def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora=False, init_method=init.xavier_normal_, input_is_parallel=True, gather_output=False, **parallel_linear_kwargs):
    if r <= 0:
        raise ValueError(f'`r` should be a positive integer value but the value passed is {r}')
    self.r[adapter_name] = r
    self.lora_alpha[adapter_name] = lora_alpha
    if lora_dropout > 0.0:
        lora_dropout_layer = nn.Dropout(p=lora_dropout)
    else:
        lora_dropout_layer = nn.Identity()
    self.lora_dropout[adapter_name] = lora_dropout_layer
    megatron_config = parallel_linear_kwargs['megatron_config']
    megatron_config.params_dtype = torch.float32
    if self.is_parallel_a:
        lora_a = self.backend.RowParallelLinear(input_size=self.in_features, output_size=r, bias=False, input_is_parallel=input_is_parallel, skip_bias_add=True, init_method=init_method, config=megatron_config)
        lora_b = nn.Linear(in_features=r, out_features=self.out_features, bias=False, dtype=torch.float32)
    else:
        lora_a = nn.Linear(in_features=self.in_features, out_features=r, bias=False, dtype=torch.float32)
        lora_b = self.backend.ColumnParallelLinear(input_size=r, output_size=self.out_features, bias=False, gather_output=gather_output, init_method=init_method, config=megatron_config)
    self.lora_A[adapter_name] = lora_a
    self.lora_B[adapter_name] = lora_b
    if use_rslora:
        self.scaling[adapter_name] = lora_alpha / r ** 0.5
    else:
        self.scaling[adapter_name] = lora_alpha / r
    if init_lora_weights:
        self.reset_lora_parameters(adapter_name, init_lora_weights)
    weight = getattr(self.get_base_layer(), 'weight', None)
    if weight is not None:
        if weight.dtype.is_floating_point or weight.dtype.is_complex:
            self.to(weight.device, dtype=weight.dtype)
        else:
            self.to(weight.device)
    self.set_adapter(self.active_adapters)