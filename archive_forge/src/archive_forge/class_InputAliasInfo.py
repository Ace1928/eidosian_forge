import collections
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, NewType, Optional, Set, Union
import torch
import torch.utils._pytree as pytree
from torch._guards import Source
from torch._subclasses import FakeTensor
from torch._subclasses.fake_tensor import is_fake
from .. import config
from .utils import strict_zip
@dataclass(frozen=True)
class InputAliasInfo:
    is_leaf: bool
    mutates_data: bool
    mutates_metadata: bool
    mutations_hidden_from_autograd: bool
    mutations_under_no_grad_or_inference_mode: bool
    mutates_storage_metadata: bool
    requires_grad: bool
    mutation_type: MutationType

    def __post_init__(self):
        if self.mutates_storage_metadata:
            assert self.mutates_metadata