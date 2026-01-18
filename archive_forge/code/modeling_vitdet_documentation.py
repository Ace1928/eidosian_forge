import collections.abc
import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput, BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_vitdet import VitDetConfig

        Returns:

        Examples:

        ```python
        >>> from transformers import VitDetConfig, VitDetBackbone
        >>> import torch

        >>> config = VitDetConfig()
        >>> model = VitDetBackbone(config)

        >>> pixel_values = torch.randn(1, 3, 224, 224)

        >>> with torch.no_grad():
        ...     outputs = model(pixel_values)

        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 768, 14, 14]
        ```