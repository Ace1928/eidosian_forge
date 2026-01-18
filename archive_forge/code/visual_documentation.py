import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from xformers.components.attention import Attention, AttentionConfig, register_attention

        Large kernel attention mechanism, as proposed in `Visual Attention Network`_, Guo et al (2022).
        The original notation is tentatively kept as is. See https://github.com/Visual-Attention-Network
        for the reference implementation

        .. Note: compared to the paper, this block contains the LKA (Large Kernel Attention)
            and the prior and posterior transformations (Conv2d and activation)

        .. _`Visual Attention Network` : https://arxiv.org/pdf/2202.09741.pdf
        