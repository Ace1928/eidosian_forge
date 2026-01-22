import argparse
import json
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List
import timm
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torch import Tensor
from transformers import AutoImageProcessor, ResNetConfig, ResNetForImageClassification
from transformers.utils import logging

        Transfer the weights of `self.src` to `self.dest` by performing a forward pass using `x` as input. Under the
        hood we tracked all the operations in both modules.
        