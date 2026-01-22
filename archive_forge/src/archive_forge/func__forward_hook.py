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
def _forward_hook(self, m, inputs: Tensor, outputs: Tensor):
    has_not_submodules = len(list(m.modules())) == 1 or isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d)
    if has_not_submodules:
        if not isinstance(m, VanLayerScaling):
            self.traced.append(m)