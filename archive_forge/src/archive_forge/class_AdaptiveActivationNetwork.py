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
class AdaptiveActivationNetwork(nn.Module):

    def __init__(self, activation_dict: ActivationDictionary, in_features: int, out_features: int, layers_info: List[int]) -> None:
        """
        Initialize the AdaptiveActivationNetwork with specific configurations for activation functions,
        input features, output features, and layer information.

        Parameters:
            activation_dict (ActivationDictionary): A dictionary containing activation functions.
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            layers_info (List[int]): A list detailing the number of neurons in each hidden layer.

        Raises:
            ValueError: If layers_info is empty or if activation_dict does not contain any valid activation functions.
        """
        super(AdaptiveActivationNetwork, self).__init__()
        if not layers_info:
            raise ValueError('The layers_info parameter must contain at least one element to define the network layers.')
        self.activations: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = activation_dict.activation_types
        if not self.activations:
            raise ValueError('The activation_dict provided does not contain any activation functions.')
        self.activation_keys: List[str] = list(self.activations.keys())
        if not self.activation_keys:
            raise ValueError('No activation functions found within the activation_dict.')
        layers: List[nn.Module] = []
        current_dim: int = in_features
        for dim in layers_info + [out_features]:
            layers.append(nn.Linear(current_dim, dim))
            current_dim = dim
        self.model: nn.Sequential = nn.Sequential(*layers)
        logger.info(f'All network layers initialized successfully within nn.Sequential with total layers: {len(self.model)}')
        self.policy_network: EnhancedPolicyNetwork = EnhancedPolicyNetwork(input_dim=current_dim, output_dim=len(self.activations), layers_info=layers_info + [out_features])
        logger.debug(f'AdaptiveActivationNetwork initialized with the following parameters: activation keys={self.activation_keys}, input features dimension={in_features}, output features dimension={out_features}, layers information={layers_info}')
        self.activation_path_log: defaultdict = defaultdict(list)

    def forward(self, x: torch.Tensor, batch_id: Optional[int]=None) -> torch.Tensor:
        """
        Conducts a forward pass through the AdaptiveActivationNetwork, processing the input tensor through each layer
        and applying dynamic activations. This method meticulously logs each step for debugging and traceability.

        Parameters:
        x (torch.Tensor): The input tensor to be processed through the network.
        batch_id (Optional[int]): An optional identifier for the batch being processed, used for logging purposes.

        Returns:
        torch.Tensor: The tensor resulting from the forward pass through the network after all transformations.

        Raises:
        RuntimeError: If any exception occurs during the forward pass, encapsulating the original exception.
        """
        logger.debug(f'Forward pass through AdaptiveActivationNetwork initiated with batch_id={batch_id}')
        layer_index: int = 0
        try:
            for layer_index, layer in enumerate(self.model[:-1]):
                logger.debug(f'Processing layer {layer_index} with input tensor dimensions: {x.shape}')
                x = nn.Linear(layer.in_features, layer.out_features)(x)
                logger.debug(f'Output tensor dimensions after DimensionSafeLinear at layer {layer_index}: {x.shape}')
                try:
                    x = self.dynamic_activation(x, batch_id)
                except Exception as e:
                    logger.error(f'Dynamic activation failed at layer {layer_index}: {e}', exc_info=True)
                    raise
                logger.debug(f'Output tensor dimensions after dynamic activation at layer {layer_index}: {x.shape}')
            final_layer_index = len(self.model) - 1
            final_layer = self.model[-1]
            logger.debug(f'Processing final layer {final_layer_index} with input tensor dimensions: {x.shape}')
            x = nn.Linear(final_layer.in_features, final_layer.out_features)(x)
            logger.debug(f'Output tensor dimensions after final layer {final_layer_index}: {x.shape}')
        except Exception as e:
            logger.error(f'An error occurred during the forward pass at layer {layer_index}: {e}', exc_info=True)
        raise RuntimeError(f'Forward pass failed due to: {e}') from e
        logger.debug(f'Final output after dynamic activation: {x}')
        return x

    def dynamic_activation(self, x: torch.Tensor, batch_id: Optional[int], deterministic: bool=False) -> torch.Tensor:
        """
        Dynamically selects and applies activation functions based on the policy network's output,
        with an option for deterministic or probabilistic selection of activations.

        Args:
            x (torch.Tensor): The input tensor to be processed through activation functions.
            batch_id (Optional[int]): An identifier for the batch, used for logging purposes.
            deterministic (bool): If True, the activation function with the highest score is selected,
                                  otherwise, a probabilistic approach is used.

        Returns:
            torch.Tensor: The tensor resulting from applying the selected activation functions,
                          with padding applied to ensure all output tensors have the same size.
        """
        activation_scores: torch.Tensor = self.policy_network(x)
        logger.debug(f'Activation scores computed for batch_id={batch_id}: {activation_scores}')
        if deterministic:
            selected_activation_idx: torch.Tensor = torch.argmax(activation_scores, dim=-1)
            logger.info(f'Deterministic selection of activation indices for batch_id={batch_id}: {selected_activation_idx}')
        else:
            activation_probs: torch.Tensor = F.softmax(activation_scores, dim=-1)
            selected_activation_idx: torch.Tensor = torch.multinomial(activation_probs, 1).squeeze(0)
            logger.info(f'Probabilistic selection of activation indices for batch_id={batch_id}: {selected_activation_idx}')
        selected_activations: List[Callable[[torch.Tensor], torch.Tensor]] = [self.activations[self.activation_keys[int(idx.long())]] for idx in selected_activation_idx.long()]
        logger.debug(f'Selected activation functions for batch_id={batch_id}: {selected_activations}')
        activated_tensors: List[torch.Tensor] = [act(x_i) for x_i, act in zip(x, selected_activations)]
        logger.debug(f'Activated tensors before padding for batch_id={batch_id}: {activated_tensors}')
        max_size: int = max((tensor.nelement() for tensor in activated_tensors), default=0)
        padded_tensors: List[torch.Tensor] = []
        for tensor in activated_tensors:
            if tensor.nelement() == 0:
                padded_tensor = torch.zeros((max_size,), dtype=tensor.dtype, device=tensor.device) if max_size > 0 else torch.tensor([], dtype=tensor.dtype, device=tensor.device)
            else:
                padding_needed = max_size - tensor.nelement()
                padded_tensor = F.pad(tensor, (0, padding_needed), 'constant', 0)
            padded_tensors.append(padded_tensor)
            logger.debug(f'Padded tensor for batch_id={batch_id}: {padded_tensor}')
        final_output = torch.stack(padded_tensors)
        logger.debug(f'Final output tensor after dynamic activation for batch_id={batch_id}: {final_output.shape}')
        return final_output