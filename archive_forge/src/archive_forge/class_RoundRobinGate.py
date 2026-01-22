import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Union
import torch
from xformers.components import Activation
from xformers.components.feedforward import (
class RoundRobinGate(torch.nn.Module):

    def __init__(self, model_dim, num_experts):
        super().__init__()
        self.model_dim = model_dim
        self.num_experts = num_experts

    def forward(self, input):
        s = input.shape[0]
        assert s % self.num_experts == 0, f'{s} % {self.num_experts} != 0'
        capacity = 2 * s // self.num_experts
        output = torch.zeros(s, self.num_experts, capacity, dtype=input.dtype, device=input.device)
        for i in range(s):
            output[i, i % self.num_experts, i // self.num_experts] = 1.0
        return (0.0, output, output.bool())