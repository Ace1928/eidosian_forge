from functools import partial
import torch.nn as nn
from einops import rearrange
from torch import _assert
from torch.nn.modules.utils import _pair
2D Image to Patch Embedding