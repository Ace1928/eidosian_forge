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

# Extending the system path to include the specific directory for module importation
sys.path.append("/home/lloyd/EVIE/Indellama3/indego")

from ActivationDictionary import ActivationDictionary

activation_dict_instance = ActivationDictionary()

from IndegoLogging import configure_logging


async def setup_logging() -> None:
    await configure_logging()


try:
    loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
except RuntimeError as e:
    if "There is no current event loop in thread" in str(e):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

if asyncio.get_event_loop().is_running():
    asyncio.run(setup_logging())
else:
    loop = asyncio.get_event_loop()
    loop.run_until_complete(setup_logging())

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class EnhancedPolicyNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layers_info: List[int],
        hyperparameters: Optional[Dict[str, float]] = None,
    ):
        super(EnhancedPolicyNetwork, self).__init__()
        if (
            input_dim <= 0
            or output_dim <= 0
            or not layers_info
            or any(dim <= 0 for dim in layers_info)
        ):
            raise ValueError("Invalid dimensions or layers information provided.")
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.layers_info: List[int] = layers_info
        self.hyperparameters: Dict[str, float] = (
            hyperparameters
            if hyperparameters is not None
            else {
                "learning_rate": 0.001,
                "regularization_factor": 0.01,
                "discount_factor": 0.99,
            }
        )
        layers: List[nn.Module] = []
        current_dim: int = input_dim
        for index, dim in enumerate(layers_info):
            linear_layer: nn.Module = nn.Linear(current_dim, dim)
            layers.append(linear_layer)
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(dim))
            current_dim = dim
        final_output_layer: nn.Module = nn.Linear(current_dim, output_dim)
        layers.append(final_output_layer)
        self.model: nn.Sequential = nn.Sequential(*layers)
        logger.debug(
            f"EnhancedPolicyNetwork initialized with input dimension {input_dim}, output dimension {output_dim}, layers {layers_info}, and initial hyperparameters {self.hyperparameters}"
        )

    def forward(
        self,
        x: torch.Tensor,
        batch_id: Optional[int] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Executes a forward pass through the EnhancedPolicyNetwork with optional determinism.

        This method processes an input tensor `x` through the network's sequential model. If the `deterministic`
        flag is set, it seeds the random number generator to ensure reproducibility. Detailed debug logging is
        provided throughout the execution to trace the computation values and any exceptions encountered.

        Parameters:
            x (torch.Tensor): The input tensor to be processed by the network.
            batch_id (Optional[int]): An optional identifier for the batch being processed, used for logging.
            deterministic (bool): If True, the random number generator is seeded to make the operation deterministic.

        Returns:
            torch.Tensor: The tensor resulting from the forward pass through the network.

        Raises:
            RuntimeError: If any exception occurs during the forward pass, it logs the error and re-raises a
                           RuntimeError with the error message encapsulated.
        """
        # Log the initiation of the forward pass with optional batch ID and determinism flag
        logger.debug(
            f"Forward pass of the EnhancedPolicyNetwork initiated with batch_id={batch_id} and deterministic={deterministic}"
        )

        try:
            # Process the input tensor through the sequential model
            x = self.model(x)
            for layer_index, layer in enumerate(self.model[:-1]):
                # Ensure that `x` is correctly processed through each layer
                x = layer(x)

            # If deterministic processing is required, set the manual seed
            if deterministic:
                torch.manual_seed(0)
                logger.debug(
                    "Random number generator seeded for deterministic operation."
                )

            # Log the final output after processing through the model
            logger.debug(
                f"Final output of the model (Q-values for each action) after processing: {x}"
            )

        except Exception as e:
            # Log the exception with detailed stack trace
            logger.error(
                f"An error occurred during the forward pass: {e}", exc_info=True
            )
            # Re-raise the exception as a RuntimeError to indicate forward pass failure
            raise RuntimeError(f"Forward pass failed due to: {e}") from e

        # Return the processed tensor
        return x


            self.activations[self.activation_keys[idx]]
            for idx in selected_activation_idx
        ]
        logger.debug(f"Selected activation functions: {selected_activations}")
        activated_tensors: List[torch.Tensor] = []
        for x_i, act in zip(x, selected_activations):
            activated_tensors.append(act(x_i))
        max_size: int = max(
            (tensor.nelement() for tensor in activated_tensors), default=0
        )
        padded_tensors: List[torch.Tensor] = []
        for tensor in activated_tensors:
            if tensor.nelement() == 0:
                if max_size > 0:
                    shape = (max_size,)
                    padded_tensor = torch.zeros(
                        shape, dtype=tensor.dtype, device=tensor.device
                    )
                else:
                    padded_tensor = torch.tensor(
                        [], dtype=tensor.dtype, device=tensor.device
                    )
            else:
                padding_needed = max_size - tensor.nelement()
                padded_tensor = F.pad(tensor, (0, padding_needed))
            padded_tensors.append(padded_tensor)
        final_output = torch.stack(padded_tensors)
        return final_output


def calculate_reward(
    current_loss: float, previous_loss: float, y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    if y_true.shape[0] != y_pred.shape[0]:
        logger.error(
            "The true labels and predicted labels arrays do not match in length."
        )
        raise ValueError(
            "The length of true labels and predicted labels must be the same."
        )
    loss_improvement: float = previous_loss - current_loss
    logger.debug(f"Calculated loss improvement: {loss_improvement}")
    precision_metric: float = precision_score(y_true, y_pred, average="macro")
    recall_metric: float = recall_score(y_true, y_pred, average="macro")
    f1_metric: float = f1_score(y_true, y_pred, average="macro")
    logger.info(
        f"Reward calculation details: Loss Improvement: {loss_improvement}, Precision: {precision_metric}, Recall: {recall_metric}, F1 Score: {f1_metric}"
    )
    reward: float = loss_improvement + 0.5 * (
        precision_metric + recall_metric + f1_metric
    )
    logger.debug(f"Constructed reward using weighted metrics: {reward}")
    return reward


def update_policy_network(
    policy_network: nn.Module,
    optimizer: torch.optim.Optimizer,
    reward: float,
    log_prob: torch.Tensor = torch.tensor(0.5, requires_grad=True),
) -> None:
    try:
        reward_tensor: torch.Tensor = torch.tensor(
            reward, requires_grad=True, device=log_prob.device
        )
        logger.debug(f"Converted reward to tensor: {reward_tensor}")
        policy_loss: torch.Tensor = -log_prob * reward_tensor
        logger.debug(f"Calculated policy loss: {policy_loss.item()}")
        optimizer.zero_grad()
        logger.debug("Reset optimizer gradients to zero.")
        policy_loss.backward()
        logger.debug("Performed backpropagation to compute gradients.")
        optimizer.step()
        logger.debug("Updated weights of the policy network.")
        logger.debug(
            f"Completed policy network update. Computed Loss: {policy_loss.item()}"
        )
    except Exception as e:
        logger.error(
            f"An error occurred during the policy network update: {e}", exc_info=True
        )
        raise RuntimeError(f"Policy network update failed due to: {e}") from e


def log_decision(
    layer_output: torch.Tensor, chosen_activation: str, reward: float
) -> None:
    try:
        layer_output_list: List[float] = layer_output.tolist()
        logger.debug(f"Converted layer output to list: {layer_output_list}")
        log_message: str = (
            f"Decision Log - Layer Output: {layer_output_list}, Chosen Activation Function: {chosen_activation}, Reward Received: {reward}"
        )
        logger.debug(f"Constructed log message: {log_message}")
        logger.info(log_message)
        logger.debug("Successfully logged the decision details at the INFO level.")
    except Exception as e:
        error_message: str = f"An error occurred while logging the decision: {e}"
        logger.error(error_message, exc_info=True)
        raise RuntimeError(f"Logging of the decision failed due to: {e}") from e


def main() -> None:
    try:
        network: AdaptiveActivationNetwork = AdaptiveActivationNetwork(
            activation_dict_instance,
            in_features=20,
            out_features=20,
            layers_info=[20, 30],
        )
        logger.info(
            "AdaptiveActivationNetwork initialized successfully with configurations: in_features=20, out_features=20, layers_info=[20, 30]"
        )
        input_tensor: torch.Tensor = torch.randn(20, 20)
        logger.debug(f"Generated random input tensor with shape {input_tensor.shape}")
        output: torch.Tensor = network(input_tensor, batch_id=1)
        logger.info(
            f"Processed input tensor through the network, resulting in output tensor with shape {output.shape}"
        )
        print("Output of the network:", output)
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}", exc_info=True)
        raise Exception(f"Failed to execute main function due to: {e}") from e


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(
            f"Critical failure in running the main function: {e}", exc_info=True
        )
