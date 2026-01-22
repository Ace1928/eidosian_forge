import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, List, Optional
import torch
from torch import fx
from torch._dynamo.output_graph import GraphCompileReason
from torch._dynamo.utils import deepcopy_to_fake_tensor, detect_fake_mode
from torch.fx.node import Node
class WrapperModule(torch.nn.Module):

    def __init__(self, submod, unwrap_singleton_tuple):
        super().__init__()
        self.submod = submod
        self.unwrap_singleton_tuple = unwrap_singleton_tuple

    def forward(self, *args):
        x = self.submod(*args)
        if self.unwrap_singleton_tuple and isinstance(x, (tuple, list)):
            return x[0]
        return x