from builtins import isinstance
import functools
import logging
from typing import Any, List, Tuple
import torch
from torch import nn
class ModelShard(nn.Module):
    """
    Wrap one shard of the model, make it possible to load parameters on the
    fly for the FW and BW pass on the given device.
    """

    def __init__(self, cpu_model_shard: nn.Module, device: torch.device, offload_device: torch.device, index: int):
        super().__init__()
        self.model_shard = cpu_model_shard
        self.index = index
        self.device = device
        torch.cuda.device(self.device)
        self.offload_device = offload_device
        self.model_shard.to(offload_device)
        self._cpu_to_gpu_stream = torch.cuda.Stream(device=self.device)
        self._gpu_to_cpu_stream = torch.cuda.Stream(device=self.device)

    def forward(self, *inputs):
        return self.model_shard(*inputs) if isinstance(inputs, tuple) else self.model_shard(inputs)

    def to(self, device: torch.device) -> 'ModelShard':
        self.model_shard.to(device)
        return self

    def train(self, mode: bool=True) -> 'ModelShard':
        self.model_shard.train(mode)
        return self

    def to_device(self) -> None:
        self.model_shard.to(device=self.device, non_blocking=True)

    def forward_load(self, non_blocking: bool=True) -> None:
        with torch.cuda.stream(self._cpu_to_gpu_stream):
            self.model_shard.to(device=self.device, non_blocking=non_blocking)

    def backward_load(self, non_blocking: bool=True) -> None:
        with torch.cuda.stream(self._cpu_to_gpu_stream):
            self.model_shard.to(self.device, non_blocking=non_blocking)

    def forward_drop(self, non_blocking: bool=True) -> None:
        with torch.cuda.stream(self._gpu_to_cpu_stream):
            self.model_shard.to(self.offload_device, non_blocking=non_blocking)

    def backward_drop(self, non_blocking: bool=True) -> None:
        with torch.cuda.stream(self._gpu_to_cpu_stream):
            self.model_shard.to(self.offload_device, non_blocking=non_blocking)