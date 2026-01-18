import math
import warnings
from typing import Any, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import gather_params_ctx
from peft.utils.other import transpose
from .config import LoraConfig
def unmerge(self) -> None:
    """
        This method unmerges all merged adapter layers from the base weights.
        """
    if not self.merged:
        warnings.warn('Already unmerged. Nothing to do.')
        return
    while len(self.merged_adapters) > 0:
        active_adapter = self.merged_adapters.pop()
        if active_adapter in self.lora_A.keys():
            self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)