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
class NameToFromModelFuncMap(dict):
    """
    A Dictionary with some additional logic to return a function that creates the correct original model.
    """

    def convert_name_to_timm(self, x: str) -> str:
        x_split = x.split('-')
        return x_split[0] + x_split[1] + '_' + ''.join(x_split[2:])

    def __getitem__(self, x: str) -> Callable[[], Tuple[nn.Module, Dict]]:
        if x not in self:
            x = self.convert_name_to_timm(x)
            val = partial(lambda: (timm.create_model(x, pretrained=True).eval(), None))
        else:
            val = super().__getitem__(x)
        return val