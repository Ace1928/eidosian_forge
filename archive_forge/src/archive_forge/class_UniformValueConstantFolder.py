import logging
import typing
from collections import Counter
from typing import Dict, Set
import torch
import torch._guards
from torch._inductor.constant_folding import ConstantFolder
from torch.multiprocessing.reductions import StorageWeakRef
from .. import config
from ..pattern_matcher import (
from .replace_random import replace_random_passes
class UniformValueConstantFolder(ConstantFolder):
    """
    Runs constant folding and replaces tensors that have a unifrom value
    with a tensor constructor call: aten.full([shape], value, ...)
    """

    def __init__(self, gm, skip_constructors=False):
        super().__init__(gm, skip_constructors)
        self.node_storages_ptrs: Dict[torch.fx.Node, int] = {}
        self.constant_data_ptrs: Dict[torch.fx.Node, StorageWeakRef] = {}

    def insertable_tensor_check(self, t: torch.Tensor) -> bool:
        return t.numel() != 0 and bool((t == t.flatten()[0]).all()) and torch._C._has_storage(t) and (t.layout == torch.strided)

    def add_node_replacement(self, node: torch.fx.Node, tensor: torch.Tensor) -> None:
        self.node_replacements[node] = tensor.flatten()[0].item()
        self.constant_data_ptrs[node] = StorageWeakRef(tensor.untyped_storage())