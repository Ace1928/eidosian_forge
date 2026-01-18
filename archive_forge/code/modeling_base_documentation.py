import json
import logging
import os
from copy import deepcopy
from typing import Optional
import torch
import torch.nn as nn
from accelerate import PartialState
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import (
from safetensors.torch import load_file as safe_load_file
from transformers import PreTrainedModel
from ..import_utils import is_npu_available, is_peft_available, is_transformers_greater_than, is_xpu_available

        Computes the reward score for a given input. The method has first to enable the adapter
        and then compute the reward score. After that the model disables the reward modeling
        adapter and enables the default ppo adapter again.
        