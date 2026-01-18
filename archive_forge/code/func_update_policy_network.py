import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict
import numpy as np
import asyncio
from typing import Dict, Any, List, cast, Tuple, Callable, Optional, Union
import sys
import torch.optim as optim
import os
from ActivationDictionary import ActivationDictionary
from IndegoLogging import configure_logging
def update_policy_network(policy_network: nn.Module, optimizer: torch.optim.Optimizer, reward: float, log_prob: torch.Tensor=torch.tensor(0.5, requires_grad=True)) -> None:
    """
    Update the policy network based on the calculated reward and the log probability of the taken action.
    This function handles the conversion of the reward to a tensor, computes the policy loss, and updates
    the policy network using backpropagation.

    Args:
    policy_network (nn.Module): The neural network model that represents the policy.
    optimizer (torch.optim.Optimizer): The optimizer used for updating the network.
    reward (float): The reward obtained from the environment.
    log_prob (torch.Tensor): The log probability of the action taken by the policy network.
    """
    try:
        reward_tensor: torch.Tensor = torch.tensor(reward, requires_grad=True, device=log_prob.device)
        logger.debug(f'Converted reward to tensor: {reward_tensor}')
        policy_loss: torch.Tensor = -log_prob * reward_tensor
        logger.debug(f'Calculated policy loss: {policy_loss.item()}')
        optimizer.zero_grad()
        logger.debug('Reset optimizer gradients to zero.')
        policy_loss.backward()
        logger.debug('Performed backpropagation to compute gradients.')
        optimizer.step()
        logger.debug('Updated weights of the policy network.')
        logger.debug(f'Completed policy network update. Computed Loss: {policy_loss.item()}')
    except Exception as e:
        logger.error(f'An error occurred during the policy network update: {e}', exc_info=True)
        raise RuntimeError(f'Policy network update failed due to: {e}') from e