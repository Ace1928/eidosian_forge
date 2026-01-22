import warnings
from typing import List, Optional
import bitsandbytes as bnb
import torch
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose
from .layer import LoraLayer

            This method unmerges all merged adapter layers from the base weights.
            