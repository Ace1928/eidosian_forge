import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_vits import VitsConfig

        labels (`torch.FloatTensor` of shape `(batch_size, config.spectrogram_bins, sequence_length)`, *optional*):
            Float values of target spectrogram. Timesteps set to `-100.0` are ignored (masked) for the loss
            computation.

        Returns:

        Example:

        ```python
        >>> from transformers import VitsTokenizer, VitsModel, set_seed
        >>> import torch

        >>> tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
        >>> model = VitsModel.from_pretrained("facebook/mms-tts-eng")

        >>> inputs = tokenizer(text="Hello - my dog is cute", return_tensors="pt")

        >>> set_seed(555)  # make deterministic

        >>> with torch.no_grad():
        ...     outputs = model(inputs["input_ids"])
        >>> outputs.waveform.shape
        torch.Size([1, 45824])
        ```
        