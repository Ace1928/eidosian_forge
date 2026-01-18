import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Union
import torch
from xformers.components import Activation
from xformers.components.feedforward import (

        A MLP variant which uses the "Mixture of Experts" paradigm, as described in Gshard_.
        xFormers uses the FairScale_ implementation under the hood.

        .. warning: Please note that most of the benefits of MoE are present in a distributed training environmentt

        .. _Gshard: https://arxiv.org/pdf/2006.16668.pdf
        .. _FairScale: https://github.com/facebookresearch/fairscale/
        