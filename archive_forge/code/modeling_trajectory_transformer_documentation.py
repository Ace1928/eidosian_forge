import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from ....modeling_utils import PreTrainedModel
from ....utils import (
from .configuration_trajectory_transformer import TrajectoryTransformerConfig

        Returns:

        Examples:

        ```python
        >>> from transformers import TrajectoryTransformerModel
        >>> import torch

        >>> model = TrajectoryTransformerModel.from_pretrained(
        ...     "CarlCochet/trajectory-transformer-halfcheetah-medium-v2"
        ... )
        >>> model.to(device)
        >>> model.eval()

        >>> observations_dim, action_dim, batch_size = 17, 6, 256
        >>> seq_length = observations_dim + action_dim + 1

        >>> trajectories = torch.LongTensor([np.random.permutation(self.seq_length) for _ in range(batch_size)]).to(
        ...     device
        ... )
        >>> targets = torch.LongTensor([np.random.permutation(self.seq_length) for _ in range(batch_size)]).to(device)

        >>> outputs = model(
        ...     trajectories,
        ...     targets=targets,
        ...     use_cache=True,
        ...     output_attentions=True,
        ...     output_hidden_states=True,
        ...     return_dict=True,
        ... )
        ```
        