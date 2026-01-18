import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from ...utils import (
from .configuration_decision_transformer import DecisionTransformerConfig

        Returns:

        Examples:

        ```python
        >>> from transformers import DecisionTransformerModel
        >>> import torch

        >>> model = DecisionTransformerModel.from_pretrained("edbeeching/decision-transformer-gym-hopper-medium")
        >>> # evaluation
        >>> model = model.to(device)
        >>> model.eval()

        >>> env = gym.make("Hopper-v3")
        >>> state_dim = env.observation_space.shape[0]
        >>> act_dim = env.action_space.shape[0]

        >>> state = env.reset()
        >>> states = torch.from_numpy(state).reshape(1, 1, state_dim).to(device=device, dtype=torch.float32)
        >>> actions = torch.zeros((1, 1, act_dim), device=device, dtype=torch.float32)
        >>> rewards = torch.zeros(1, 1, device=device, dtype=torch.float32)
        >>> target_return = torch.tensor(TARGET_RETURN, dtype=torch.float32).reshape(1, 1)
        >>> timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        >>> attention_mask = torch.zeros(1, 1, device=device, dtype=torch.float32)

        >>> # forward pass
        >>> with torch.no_grad():
        ...     state_preds, action_preds, return_preds = model(
        ...         states=states,
        ...         actions=actions,
        ...         rewards=rewards,
        ...         returns_to_go=target_return,
        ...         timesteps=timesteps,
        ...         attention_mask=attention_mask,
        ...         return_dict=False,
        ...     )
        ```