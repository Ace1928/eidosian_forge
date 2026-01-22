from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional, Union
from peft.config import PeftConfig
from peft.utils import PeftType
@dataclass
class LoftQConfig:
    """
    This is the sub-configuration class to store the configuration of a [`LoraModel`].

    Args:
        bits_pattern (`dict`): The mapping from layer names or regexp expression to bits which are different from the
            default bits specified by `bits`. For example, `{model.decoder.layers.0.encoder_attn.k_proj: 2`}.
        bits (`int`): Quantization bits for LoftQ.
        iter (`int`): Alternating iterations for LoftQ.
        fake (`bool`): True: use fp16/fp32; used for first time to save weights. False: use bitsandbytes 4bit linear
            models. weights can't be saved. Recommend to set to True, save the weights and load the saved weights in 4
            bits.
    """
    loftq_bits: int = field(default=4, metadata={'help': 'Quantization bits for LoftQ'})
    loftq_iter: int = field(default=1, metadata={'help': 'Alternating iterations for LoftQ'})