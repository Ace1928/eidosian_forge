import math
import re
from collections import OrderedDict
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPTBigCodeConfig, PretrainedConfig

    Map the state_dict of a flash_attn model to be Huggingface BigCode compatible.

    This function is meant to be the inverse of remap_state_dict_hf_bigcode.
    