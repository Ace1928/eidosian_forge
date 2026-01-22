from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple
import torch.nn.functional as F
from torch import nn, Tensor
from ..ops.misc import Conv2dNormActivation
from ..utils import _log_api_usage_once

        Computes the FPN for a set of feature maps.

        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.

        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from the highest resolution first.
        