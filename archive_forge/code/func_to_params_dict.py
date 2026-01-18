import argparse
import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from classy_vision.models.regnet import RegNet, RegNetParams
from huggingface_hub import cached_download, hf_hub_url
from torch import Tensor
from vissl.models.model_helpers import get_trunk_forward_outputs
from transformers import AutoImageProcessor, RegNetConfig, RegNetForImageClassification, RegNetModel
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
def to_params_dict(dict_with_modules):
    params_dict = OrderedDict()
    for name, module in dict_with_modules.items():
        for param_name, param in module.state_dict().items():
            params_dict[f'{name}.{param_name}'] = param
    return params_dict