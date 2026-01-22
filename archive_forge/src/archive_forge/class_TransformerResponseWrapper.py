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
class TransformerResponseWrapper(nn.Module):
    """
    Wrap transformer response.

    Pushes input through transformer and MLP.
    """

    def __init__(self, transformer, hdim):
        super(TransformerResponseWrapper, self).__init__()
        dim = transformer.out_dim
        self.transformer = transformer
        self.mlp = nn.Sequential(nn.Linear(dim, hdim), nn.ReLU(), nn.Linear(hdim, dim))

    def forward(self, *args):
        """
        Forward pass.
        """
        return self.mlp(self.transformer(*args))