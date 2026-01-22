import argparse
import json
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import timm
import torch
import torch.nn as nn
from classy_vision.models.regnet import RegNet, RegNetParams, RegNetY32gf, RegNetY64gf, RegNetY128gf
from huggingface_hub import cached_download, hf_hub_url
from torch import Tensor
from vissl.models.model_helpers import get_trunk_forward_outputs
from transformers import AutoImageProcessor, RegNetConfig, RegNetForImageClassification, RegNetModel
from transformers.utils import logging
class FakeRegNetVisslWrapper(nn.Module):
    """
    Fake wrapper for RegNet that mimics what vissl does without the need to pass a config file.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        feature_blocks: List[Tuple[str, nn.Module]] = []
        feature_blocks.append(('conv1', model.stem))
        for k, v in model.trunk_output.named_children():
            assert k.startswith('block'), f'Unexpected layer name {k}'
            block_index = len(feature_blocks) + 1
            feature_blocks.append((f'res{block_index}', v))
        self._feature_blocks = nn.ModuleDict(feature_blocks)

    def forward(self, x: Tensor):
        return get_trunk_forward_outputs(x, out_feat_keys=None, feature_blocks=self._feature_blocks)