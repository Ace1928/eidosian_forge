import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_seamless_m4t_v2 import SeamlessM4Tv2Config
def _indices_to_subwords(self, input_ids):
    """
        Returns the corresponding text string for each input id.
        """
    if not hasattr(self.generation_config, 'id_to_text'):
        raise ValueError("This model generation config doesn't have a `id_to_text` key which maps\n                token ids to subwords. Make sure to load the right generation config.")
    batch_size, sequence_len = input_ids.shape
    subwords_batch = []
    for batch_id in range(batch_size):
        subwords = []
        for i in range(sequence_len):
            subword = self.generation_config.id_to_text.get(str(input_ids[batch_id, i].item()))
            subwords.append(str(subword))
        subwords_batch.append(subwords)
    return subwords_batch