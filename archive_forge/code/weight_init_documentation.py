import logging
import math
from enum import Enum
from typing import Callable
import torch
import torch.nn as nn
from torch.nn.init import (

    ViT weight initialization, original timm impl (for reproducibility).

    See DeepNet_ for all the DeepNorm specific codepaths
    