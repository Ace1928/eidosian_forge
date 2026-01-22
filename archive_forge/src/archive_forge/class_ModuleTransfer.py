import argparse
import json
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List
import torch
import torch.nn as nn
from huggingface_hub import cached_download, hf_hub_download
from torch import Tensor
from transformers import AutoImageProcessor, VanConfig, VanForImageClassification
from transformers.models.deprecated.van.modeling_van import VanLayerScaling
from transformers.utils import logging
@dataclass
class ModuleTransfer:
    src: nn.Module
    dest: nn.Module
    verbose: int = 0
    src_skip: List = field(default_factory=list)
    dest_skip: List = field(default_factory=list)

    def __call__(self, x: Tensor):
        """
        Transfer the weights of `self.src` to `self.dest` by performing a forward pass using `x` as input. Under the
        hood we tracked all the operations in both modules.
        """
        dest_traced = Tracker(self.dest)(x).parametrized
        src_traced = Tracker(self.src)(x).parametrized
        src_traced = list(filter(lambda x: type(x) not in self.src_skip, src_traced))
        dest_traced = list(filter(lambda x: type(x) not in self.dest_skip, dest_traced))
        if len(dest_traced) != len(src_traced):
            raise Exception(f'Numbers of operations are different. Source module has {len(src_traced)} operations while destination module has {len(dest_traced)}.')
        for dest_m, src_m in zip(dest_traced, src_traced):
            dest_m.load_state_dict(src_m.state_dict())
            if self.verbose == 1:
                print(f'Transfered from={src_m} to={dest_m}')