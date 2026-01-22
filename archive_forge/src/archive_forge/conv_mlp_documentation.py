import math
from dataclasses import dataclass
from typing import Optional
import torch.nn as nn
from xformers.components import Activation, build_activation
from xformers.components.feedforward import Feedforward, FeedforwardConfig
from . import register_feedforward

    A Convolutional feed-forward network, as proposed in VAN_ (Vision Attention Network, Guo et al.)

    .. _VAN: https://arxiv.org/pdf/2202.09741.pdf
    