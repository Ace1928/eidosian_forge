import unittest
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
    """Asynchronously sets up logging configurations using the IndegoLogging module."""
    await configure_logging()


# Establishing the event loop for asynchronous operations
try:
    loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
except RuntimeError as e:
    if "There is no current event loop in thread" in str(e):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

# Running the logging setup based on the state of the event loop
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
    ) -> None:
        """
        Initializes the EnhancedPolicyNetwork with specified dimensions and layer information, including optional hyperparameters.

        This constructor method meticulously sets up a neural network architecture based on the provided input dimension, output dimension,
        and information about the layers. It also allows for the specification of hyperparameters which can include learning rate, regularization factor,
        and discount factor among others. The method ensures that all parameters are validated to prevent any erroneous network configuration.

        The neural network architecture is constructed using the PyTorch framework, leveraging its powerful and flexible modules for building deep learning models.
        The architecture consists of a series of fully connected linear layers, each followed by a rectified linear unit (ReLU) activation function.
        The dimensions of each layer are determined by the `layers_info` parameter, which specifies the number of neurons in each hidden layer.

        The linear layers are created using the `nn.Linear` module from PyTorch, which performs an affine transformation on the input tensor.
        The ReLU activation function is applied after each linear layer using the `nn.ReLU` module, introducing non-linearity into the network.

        Batch normalization is also applied after each ReLU activation using the `nn.BatchNorm1d` module. Batch normalization helps to stabilize the training process
        by normalizing the activations of each layer, reducing the internal covariate shift and allowing for faster convergence.

        The hyperparameters dictionary allows for customization of the learning process, with default values provided for the learning rate, regularization factor,
        and discount factor. These hyperparameters can be adjusted to fine-tune the behavior of the network during training.

        Detailed logging is used throughout the initialization process to trace the network configuration and any potential errors or exceptions that may occur.
        The logging statements provide valuable information for debugging and monitoring the initialization of the EnhancedPolicyNetwork.

        Args:
            input_dim (int): The dimensionality of the input data.
            output_dim (int): The dimensionality of the output data.
            layers_info (List[int]): A list detailing the number of neurons in each hidden layer.
            hyperparameters (Optional[Dict[str, float]]): A dictionary containing learning parameters such as learning rate, regularization factor, etc.

        Raises:
            ValueError: If any of the dimensions or layer information is invalid, specifically if they are non-positive or not provided.

        Returns:
            None
        """
        super(EnhancedPolicyNetwork, self).__init__()

        # Validate input parameters
        if input_dim <= 0:
            logger.error(
                f"Invalid input dimension provided: {input_dim}. Input dimension must be a positive integer."
            )
            raise ValueError(
                f"Invalid input dimension provided: {input_dim}. Input dimension must be a positive integer."
            )

        if output_dim <= 0:
            logger.error(
                f"Invalid output dimension provided: {output_dim}. Output dimension must be a positive integer."
            )
            raise ValueError(
                f"Invalid output dimension provided: {output_dim}. Output dimension must be a positive integer."
            )

        if not layers_info:
            logger.error(
                "No layer information provided. At least one hidden layer must be specified."
            )
            raise ValueError(
                "No layer information provided. At least one hidden layer must be specified."
            )

        if any(layer_dim <= 0 for layer_dim in layers_info):
            logger.error(
                f"Invalid layer dimensions provided: {layers_info}. All layer dimensions must be positive integers."
            )
            raise ValueError(
                f"Invalid layer dimensions provided: {layers_info}. All layer dimensions must be positive integers."
            )

        # Store input parameters
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.layers_info: List[int] = layers_info

        # Set default hyperparameters if not provided
        if hyperparameters is None:
            self.hyperparameters: Dict[str, float] = {
                "learning_rate": 0.001,
                "regularization_factor": 0.01,
                "discount_factor": 0.99,
            }
        else:
            self.hyperparameters: Dict[str, float] = hyperparameters

        # Log the initialization details
        logger.debug(
            f"Initializing EnhancedPolicyNetwork with the following configuration:\n"
            f"  Input Dimension: {self.input_dim}\n"
            f"  Output Dimension: {self.output_dim}\n"
            f"  Layer Information: {self.layers_info}\n"
            f"  Hyperparameters: {self.hyperparameters}"
        )

        # Construct the neural network architecture
        layers: List[nn.Module] = []
        prev_dim: int = self.input_dim

        # Iterate over each hidden layer dimension
        for layer_dim in self.layers_info:
            # Create a linear layer with the previous dimension as input and the current dimension as output
            linear_layer: nn.Linear = nn.Linear(prev_dim, layer_dim)
            layers.append(linear_layer)
            logger.debug(f"Added linear layer: {linear_layer}")

            # Apply ReLU activation function after the linear layer
            relu_layer: nn.ReLU = nn.ReLU()
            layers.append(relu_layer)
            logger.debug(f"Added ReLU activation: {relu_layer}")

            # Apply batch normalization after the ReLU activation
            batch_norm_layer: nn.BatchNorm1d = nn.BatchNorm1d(layer_dim)
            layers.append(batch_norm_layer)
            logger.debug(f"Added batch normalization: {batch_norm_layer}")

            # Update the previous dimension for the next iteration
            prev_dim = layer_dim

        # Create the final output layer
        output_layer: nn.Linear = nn.Linear(prev_dim, self.output_dim)
        layers.append(output_layer)
        logger.debug(f"Added output layer: {output_layer}")

        # Create the sequential model by passing the layers as arguments
        self.model: nn.Sequential = nn.Sequential(*layers)
        logger.debug(f"Created sequential model: {self.model}")

        # Log the successful initialization of the EnhancedPolicyNetwork
        logger.info("EnhancedPolicyNetwork initialized successfully.")

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


class AdaptiveActivationNetwork(nn.Module):
    def __init__(
        self,
        activation_dict: ActivationDictionary,
        in_features: int,
        out_features: int,
        layers_info: List[int],
    ) -> None:
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
            raise ValueError(
                "The layers_info parameter must contain at least one element to define the network layers."
            )
        self.activations: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = (
            activation_dict.activation_types
        )
        if not self.activations:
            raise ValueError(
                "The activation_dict provided does not contain any activation functions."
            )
        self.activation_keys: List[str] = list(self.activations.keys())
        if not self.activation_keys:
            raise ValueError(
                "No activation functions found within the activation_dict."
            )

        # Constructing the network layers using nn.Sequential for automatic handling of input and output dimensions
        layers: List[nn.Module] = []
        current_dim: int = in_features
        for dim in layers_info + [out_features]:
            layers.append(nn.Linear(current_dim, dim))
            current_dim = dim
        self.model: nn.Sequential = nn.Sequential(*layers)

        logger.info(
            f"All network layers initialized successfully within nn.Sequential with total layers: {len(self.model)}"
        )

        # Initializing the policy network with the output dimension set to the number of activations
        self.policy_network: EnhancedPolicyNetwork = EnhancedPolicyNetwork(
            input_dim=current_dim,
            output_dim=len(self.activations),
            layers_info=layers_info + [out_features],
        )

        logger.debug(
            f"AdaptiveActivationNetwork initialized with the following parameters: activation keys={self.activation_keys}, input features dimension={in_features}, output features dimension={out_features}, layers information={layers_info}"
        )

        self.activation_path_log: defaultdict = defaultdict(list)

    def forward(self, x: torch.Tensor, batch_id: Optional[int] = None) -> torch.Tensor:
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
        logger.debug(
            f"Forward pass through AdaptiveActivationNetwork initiated with batch_id={batch_id}"
        )

        # Attempt to process each layer and apply dynamic activations
        try:
            # Iterate over all layers except the final one
            for layer_index, layer in enumerate(self.model[:-1]):
                logger.debug(
                    f"Processing layer {layer_index} with input tensor dimensions: {x.shape}"
                )

                # Apply a dimension-safe linear transformation
                x = nn.Linear(layer.in_features, layer.out_features)(x)

                logger.debug(
                    f"Output tensor dimensions after DimensionSafeLinear at layer {layer_index}: {x.shape}"
                )

                # Apply dynamic activation function
                try:
                    x = self.dynamic_activation(x, batch_id)
                except Exception as e:
                    logger.error(
                        f"Dynamic activation failed at layer {layer_index}: {e}",
                        exc_info=True,
                    )
                    raise

                logger.debug(
                    f"Output tensor dimensions after dynamic activation at layer {layer_index}: {x.shape}"
                )

            # Process the final layer
            final_layer_index = len(self.model) - 1
            final_layer = self.model[-1]
            logger.debug(
                f"Processing final layer {final_layer_index} with input tensor dimensions: {x.shape}"
            )

            # Apply a dimension-safe linear transformation to the final layer
            x = nn.Linear(final_layer.in_features, final_layer.out_features)(x)

            logger.debug(
                f"Output tensor dimensions after final layer {final_layer_index}: {x.shape}"
            )

        except Exception as e:
            # Log and re-raise any exceptions that occur during the forward pass
            logger.error(
                f"An error occurred during the forward pass at layer {layer_index if 'layer_index' in locals() else 'unknown'}: {e}",
                exc_info=True,
            )
            raise RuntimeError(f"Forward pass failed due to: {e}") from e

        logger.debug(f"Final output after dynamic activation: {x}")
        return x

    def dynamic_activation(
        self, x: torch.Tensor, batch_id: Optional[int], deterministic: bool = False
    ) -> torch.Tensor:
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
        # Compute activation scores using the policy network
        activation_scores: torch.Tensor = self.policy_network(x)
        logger.debug(
            f"Activation scores computed for batch_id={batch_id}: {activation_scores}"
        )

        # Select activation indices either deterministically or probabilistically
        if deterministic:
            selected_activation_idx: torch.Tensor = torch.argmax(
                activation_scores, dim=-1
            )
            logger.info(
                f"Deterministic selection of activation indices for batch_id={batch_id}: {selected_activation_idx}"
            )
        else:
            activation_probs: torch.Tensor = F.softmax(activation_scores, dim=-1)
            selected_activation_idx: torch.Tensor = torch.multinomial(
                activation_probs, 1
            ).squeeze(0)
            logger.info(
                f"Probabilistic selection of activation indices for batch_id={batch_id}: {selected_activation_idx}"
            )

        # Retrieve the corresponding activation functions
        selected_activations: List[Callable[[torch.Tensor], torch.Tensor]] = [
            self.activations[self.activation_keys[idx.item()]]
            for idx in selected_activation_idx.long()
        ]
        logger.debug(
            f"Selected activation functions for batch_id={batch_id}: {selected_activations}"
        )

        # Apply the selected activation functions to the input tensor
        activated_tensors: List[torch.Tensor] = [
            act(x_i) for x_i, act in zip(x, selected_activations)
        ]
        logger.debug(
            f"Activated tensors before padding for batch_id={batch_id}: {activated_tensors}"
        )

        # Determine the maximum size among the activated tensors
        max_size: int = max(
            (tensor.nelement() for tensor in activated_tensors), default=0
        )

        # Pad tensors to have the same number of elements
        padded_tensors: List[torch.Tensor] = []
        for tensor in activated_tensors:
            if tensor.nelement() == 0:
                padded_tensor = (
                    torch.zeros((max_size,), dtype=tensor.dtype, device=tensor.device)
                    if max_size > 0
                    else torch.tensor([], dtype=tensor.dtype, device=tensor.device)
                )
            else:
                padding_needed = max_size - tensor.nelement()
                padded_tensor = F.pad(tensor, (0, padding_needed), "constant", 0)
            padded_tensors.append(padded_tensor)
            logger.debug(f"Padded tensor for batch_id={batch_id}: {padded_tensor}")

        # Stack the padded tensors into a single tensor
        final_output = torch.stack(padded_tensors)
        logger.debug(
            f"Final output tensor after dynamic activation for batch_id={batch_id}: {final_output.shape}"
        )

        return final_output


def calculate_reward(
    current_loss: float,
    previous_loss: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calculate the reward based on the improvement in loss and the combined metrics of precision,
    recall, and F1 score. This function ensures that the true and predicted labels have the same length,
    computes the improvement in loss, and calculates the precision, recall, and F1 score to determine
    the reward.

    Args:
    current_loss (float): The current loss after a model update.
    previous_loss (float): The loss before the model update.
    y_true (np.ndarray): The true labels.
    y_pred (np.ndarray): The predicted labels by the model.

    Returns:
    float: The calculated reward based on the specified metrics.
    """
    if y_true.shape[0] != y_pred.shape[0]:
        logger.error(
            "The true labels and predicted labels arrays do not match in length."
        )
        raise ValueError(
            "The length of true labels and predicted labels must be the same."
        )

    loss_improvement: float = previous_loss - current_loss
    logger.debug(f"Calculated loss improvement: {loss_improvement}")

    metrics: Dict[str, float] = {
        "precision": float(precision_score(y_true, y_pred, average="macro")),
        "recall": float(recall_score(y_true, y_pred, average="macro")),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    logger.info(
        f"Reward calculation details: Loss Improvement: {loss_improvement}, Metrics: {metrics}"
    )

    reward: float = loss_improvement + 0.5 * sum(metrics.values())
    logger.debug(f"Constructed reward using weighted metrics: {reward}")

    return reward


def update_policy_network(
    policy_network: nn.Module,
    optimizer: torch.optim.Optimizer,
    reward: float,
    log_prob: torch.Tensor = torch.tensor(0.5, requires_grad=True),
) -> None:
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
            f"An error occurred during the policy network update: {e}",
            exc_info=True,
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
