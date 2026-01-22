import math
from typing import Dict, Tuple, Optional, Union
import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from parlai.core.torch_generator_agent import TorchGeneratorModel
from parlai.utils.misc import warn_once
from parlai.utils.torch import neginf, PipelineHelper
def _create_selfattn_mask(self, x):
    bsz = x.size(0)
    time = x.size(1)
    mask = torch.tril(x.new(time, time).fill_(1))
    mask = mask.unsqueeze(0).expand(bsz, -1, -1)
    return mask