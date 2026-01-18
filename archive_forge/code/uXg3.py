import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Callable, List, Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle
import signal
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
MD_DIRECTORY = "/home/lloyd/Downloads/gpt_chats"
ROOT_DIR = "/media/lloyd/Aurora_M2/dandata/"
PROCESSED_DATA_DIR = "/media/lloyd/Aurora_M2/dandata/processed_data/"
SYNTHETIC_DATA_DIR = "/media/lloyd/Aurora_M2/dandata/synthetic_data/"
UNTRAINED_MODEL_PATH = "/media/lloyd/Aurora_M2/dandata/untrained_dan_model.pkl"
REGRESSION_TRAINED_MODEL_PATH = (
    "/media/lloyd/Aurora_M2/dandata/regression_trained_dan_model.pkl"
)
TEXT_TRAINED_MODEL_PATH = "/media/lloyd/Aurora_M2/dandata/text_trained_dan_model.pkl"
OUTPUT_DIR = "/media/lloyd/Aurora_M2/dandata/outputs/"
INPUT_SIZE = 1024
SYNTHETIC_DATA_SIZE = 500000
TEST_SIZE = 0.2
RANDOM_STATE = 42
LR = 0.001
WEIGHT_DECAY = 0.0001
SCALE_OUTPUT = True
VECTORIZER = TfidfVectorizer(max_features=INPUT_SIZE)
BATCH_SIZE = 100
TARGET_LOSS = 0.01
TARGET_RMSE = 0.1
TARGET_MAPE = 0.05
MAX_EPOCHS = 1000


print("Constants defined.")
logger.info("Constants defined.")


class AdaNorm(nn.Module):
    """
    Adaptive Normalization Layer.

    This layer normalizes the input tensor by its mean and standard deviation,
    and then scales and shifts the normalized tensor using learnable parameters.
    It includes options for different normalization strategies and adaptive
    mechanisms to enhance robustness and flexibility.

    Args:
        num_features (int): Number of features in the input tensor.
        eps (float): A small value to avoid division by zero.
        adaptive (bool): If True, use adaptive normalization based on running statistics.
        momentum (float): Momentum for updating running statistics.

    Attributes:
        scale (nn.Parameter): Learnable scaling parameter.
        shift (nn.Parameter): Learnable shifting parameter.
        running_mean (torch.Tensor): Running mean for adaptive normalization.
        running_var (torch.Tensor): Running variance for adaptive normalization.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        adaptive: bool = True,
        momentum: float = 0.1,
    ):
        """
        Initialize the AdaNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float): A small value to avoid division by zero.
            adaptive (bool): If True, use adaptive normalization based on running statistics.
            momentum (float): Momentum for updating running statistics.
        """
        super(AdaNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.adaptive = adaptive
        self.momentum = momentum

        self.scale = nn.Parameter(torch.ones(num_features))
        self.shift = nn.Parameter(torch.zeros(num_features))

        if self.adaptive:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the AdaNorm layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized, scaled, and shifted tensor.
        """
        if self.adaptive and self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)

            self.running_mean = (
                self.momentum * mean + (1 - self.momentum) * self.running_mean
            )
            self.running_var = (
                self.momentum * var + (1 - self.momentum) * self.running_var
            )
        else:
            mean = self.running_mean if self.adaptive else x.mean(dim=0)
            var = self.running_var if self.adaptive else x.var(dim=0, unbiased=False)

        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * normalized + self.shift

    def extra_repr(self) -> str:
        """
        Extra representation for the AdaNorm layer.

        Returns:
            str: String representation of the AdaNorm layer.
        """
        return (
            f"num_features={self.num_features}, eps={self.eps}, "
            f"adaptive={self.adaptive}, momentum={self.momentum}"
        )


class DynamicActivationNeuron(nn.Module):
    """
    Dynamic Activation Neuron class.

    This class represents a dynamic activation neuron that can learn and apply
    different activation functions and neuron types to the input data.

    Args:
        input_size (int): The size of the input features.

    Attributes:
        input_size (int): The size of the input features.
        activation_functions (list): List of activation functions.
        neuron_types (list): List of neuron types.
        activation_weights (nn.Parameter): Learnable weights for activation functions.
        neuron_type_weights (nn.Parameter): Learnable weights for neuron types.
        layer_norm (nn.LayerNorm): Layer normalization module.
        batch_norm (nn.BatchNorm1d): Batch normalization module.
        adanorm (AdaNorm): Adaptive normalization module.
        param (nn.Parameter): Learnable scaling parameter.
        basis_function (nn.Linear): Linear transformation layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: Optional[int] = None,
        scale_output: bool = True,
        activation_functions: Optional[
            List[Callable[[torch.Tensor], torch.Tensor]]
        ] = None,
        neuron_types: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
    ):
        """
        Initialize the DynamicActivationNeuron class.

        Args:
            input_size (int): The size of the input features.
            output_size (Optional[int]): The size of the output features. If None, it defaults to input_size.
            scale_output (bool): Whether to scale the output by a learnable parameter.
            activation_functions (Optional[List[Callable[[torch.Tensor], torch.Tensor]]]): List of activation functions to use.
            neuron_types (Optional[List[Callable[[torch.Tensor], torch.Tensor]]]): List of neuron type specific processing functions to use.
        """
        super(DynamicActivationNeuron, self).__init__()
        self.input_size = input_size
        self.output_size = output_size if output_size is not None else input_size
        self.scale_output = scale_output

        # Define the basis function as a linear transformation
        self.basis_function = nn.Linear(self.input_size, self.output_size)
        nn.init.kaiming_uniform_(self.basis_function.weight, nonlinearity="relu")

        # Define the learnable scaling parameter if scale_output is True
        self.param = nn.Parameter(torch.tensor(1.0)) if scale_output else None

        # Define the default activation functions based on the Kolmogorov-Arnold representation theorem
        self.activation_functions = (
            activation_functions or self.default_activation_functions()
        )

        # Define the default neuron types based on the Kolmogorov-Arnold representation theorem
        self.neuron_types = neuron_types or self.default_neuron_types()

        # Define learnable weights for the activation functions and neuron types
        self.activation_weights = nn.Parameter(
            torch.randn(len(self.activation_functions), dtype=torch.float)
        )
        self.neuron_type_weights = nn.Parameter(
            torch.randn(len(self.neuron_types), dtype=torch.float)
        )

        # Define normalization layers
        self.layer_norm = nn.LayerNorm(self.output_size)
        self.batch_norm = nn.BatchNorm1d(self.output_size)
        self.adanorm = AdaNorm(self.output_size)

        print(
            f"DynamicActivationNeuron initialized with input_size={self.input_size}, output_size={self.output_size}, scale_output={self.scale_output}"
        )
        logger.info(
            f"DynamicActivationNeuron initialized with input_size={self.input_size}, output_size={self.output_size}, scale_output={self.scale_output}"
        )

    def _ensure_tensor(self, x):
        """
        Ensure the input is a tensor. Convert if necessary.

        Args:
            x: Input data which can be a numpy array, list, or tensor.

        Returns:
            torch.Tensor: Converted tensor.
        """
        if isinstance(x, np.ndarray):
            print("Converting numpy array to tensor.")
            logger.info("Converting numpy array to tensor.")
            return torch.from_numpy(x).float()
        elif isinstance(x, list):
            print("Converting list to tensor.")
            logger.info("Converting list to tensor.")
            return torch.tensor(x, dtype=torch.float)
        elif not isinstance(x, torch.Tensor):
            raise TypeError(
                f"Unsupported input type: {type(x)}. Expected numpy.ndarray, list, or torch.Tensor."
            )
        print("Input is already a tensor.")
        logger.info("Input is already a tensor.")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DynamicActivationNeuron.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after processing.
        """
        print("Starting forward pass.")
        logger.info("Starting forward pass.")

        # Ensure input is a tensor and convert to float
        x = self._ensure_tensor(x).float()
        original_shape = x.shape
        print(f"Original shape: {original_shape}")
        logger.info(f"Original shape: {original_shape}")

        # Flatten input to match input size
        x = x.view(-1, self.input_size)
        print(f"Shape after flattening: {x.shape}")
        logger.info(f"Shape after flattening: {x.shape}")

        # Apply basis function
        x = self.basis_function(x)
        print(f"Shape after basis function: {x.shape}")
        logger.info(f"Shape after basis function: {x.shape}")

        # Apply normalization
        x = self.apply_normalization(x)
        print(f"Shape after normalization: {x.shape}")
        logger.info(f"Shape after normalization: {x.shape}")

        try:
            # Apply learned activation functions
            x = self.apply_learned_activation(x)
            print(f"Shape after learned activation: {x.shape}")
            logger.info(f"Shape after learned activation: {x.shape}")

            # Apply learned neuron types
            x = self.apply_learned_neuron_type(x)
            print(f"Shape after learned neuron type: {x.shape}")
            logger.info(f"Shape after learned neuron type: {x.shape}")
        except Exception as e:
            logger.error(f"Error in processing: {e}")
            print(f"Error in processing: {e}")
            raise RuntimeError(f"Processing failed: {e}")

        # Apply scaling parameter if defined
        if self.param is not None:
            x = self.param * x
            print(f"Shape after applying param: {x.shape}")
            logger.info(f"Shape after applying param: {x.shape}")

        # Calculate final shape dynamically
        final_shape = list(original_shape[:-1]) + [x.shape[-1]]
        print(f"Final shape to be reshaped to: {final_shape}")
        logger.info(f"Final shape to be reshaped to: {final_shape}")

        # Reshape to final shape
        x = x.view(*final_shape)
        print(f"Final output shape: {x.shape}")
        logger.info(f"Final output shape: {x.shape}")

        print("Forward pass completed.")
        logger.info("Forward pass completed.")

        # Clamp output to avoid extreme values
        return x.clamp(min=-1e6, max=1e6)

    def apply_normalization(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LayerNorm, BatchNorm, and AdaNorm to the input tensor.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Normalized tensor.
        """
        print("Applying normalization.")
        logger.info("Applying normalization.")
        try:
            # Apply LayerNorm
            x = self.layer_norm(x)
            print(f"LayerNorm applied: {x}")
            logger.info(f"LayerNorm applied: {x}")

            # Apply BatchNorm1d based on tensor dimensions
            if x.dim() == 2:  # BatchNorm1d for 2D input
                x = self.batch_norm(x)
                print(f"BatchNorm1d applied for 2D input: {x}")
                logger.info(f"BatchNorm1d applied for 2D input: {x}")
            elif x.dim() > 2:  # Handle higher dimensional inputs
                # Flatten to apply BatchNorm1d and then reshape back
                x_flat = x.view(-1, x.size(-1))
                x_norm = self.batch_norm(x_flat).view(x.size())
                x = x_norm
                print(f"BatchNorm1d applied for higher dimensional input: {x}")
                logger.info(f"BatchNorm1d applied for higher dimensional input: {x}")
            else:
                warning_msg = (
                    f"Unsupported tensor dimensionality: {x.dim()}. Skipping BatchNorm."
                )
                print(warning_msg)
                logger.warning(warning_msg)

            # Apply AdaNorm
            x = self.adanorm(x)
            print(f"AdaNorm applied: {x}")
            logger.info(f"AdaNorm applied: {x}")

        except Exception as e:
            error_msg = (
                f"Error in normalization: {e}. Continuing without normalization."
            )
            print(error_msg)
            logger.warning(error_msg)

        print("Normalization completed.")
        logger.info("Normalization completed.")
        return x

    def apply_learned_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply learned activation functions to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying activation functions.
        """
        print("Applying learned activation functions.")
        logger.info("Applying learned activation functions.")

        if not self.activation_functions:
            print("No activation functions found. Returning input tensor.")
            logger.info("No activation functions found. Returning input tensor.")
            return x

        try:
            activation_weights = F.softmax(self.activation_weights, dim=0)
            print(f"Activation weights: {activation_weights}")
            logger.info(f"Activation weights: {activation_weights}")

            activation_output = sum(
                weight * activation(x)
                for weight, activation in zip(
                    activation_weights, self.activation_functions
                )
            )

            print("Learned activation functions applied.")
            logger.info("Learned activation functions applied.")

            return activation_output.clamp(min=-1e6, max=1e6)

        except Exception as e:
            logger.error(f"Error in learned activation: {e}")
            print(f"Error in learned activation: {e}")
            raise RuntimeError(f"Activation processing failed: {e}")

    def apply_learned_neuron_type(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply learned neuron types to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying neuron types.
        """
        print("Applying learned neuron types.")
        logger.info("Applying learned neuron types.")

        if not self.neuron_types:
            print("No neuron types found. Returning input tensor.")
            logger.info("No neuron types found. Returning input tensor.")
            return x

        try:
            neuron_type_weights = F.softmax(self.neuron_type_weights, dim=0)
            print(f"Neuron type weights: {neuron_type_weights}")
            logger.info(f"Neuron type weights: {neuron_type_weights}")

            neuron_type_output = sum(
                weight * neuron(x)
                for weight, neuron in zip(neuron_type_weights, self.neuron_types)
            )

            print("Learned neuron types applied.")
            logger.info("Learned neuron types applied.")

            return neuron_type_output.clamp(min=-1e6, max=1e6)

        except Exception as e:
            error_msg = f"Error in neuron type processing: {e}"
            logger.error(error_msg)
            print(error_msg)
            raise RuntimeError(f"Neuron type processing failed: {e}")

    @staticmethod
    def default_activation_functions() -> List[Callable[[torch.Tensor], torch.Tensor]]:
        """
        Return the default activation functions based on the Kolmogorov-Arnold representation theorem.

        Returns:
        List[Callable[[torch.Tensor], torch.Tensor]]: List of default activation functions.
        """
        return [
            F.relu,
            torch.sigmoid,
            torch.tanh,
            lambda x: (x > 0).float(),
            lambda x: x**2 + x,
            torch.sin,
            lambda x: x * torch.sigmoid(x),
            lambda x: x / (1 + torch.abs(x)),
            lambda x: x * torch.sigmoid(x),
            lambda x: x * torch.tanh(F.softplus(x)),
        ]

    @staticmethod
    def default_neuron_types() -> List[Callable[[torch.Tensor], torch.Tensor]]:
        """
        Return the default neuron types based on the Kolmogorov-Arnold representation theorem.

        Returns:
        List[Callable[[torch.Tensor], torch.Tensor]]: List of default neuron types.
        """
        return [
            lambda x: (x > 0).float(),
            lambda x: x,
            lambda x: x**2 + x,
            torch.sin,
            lambda x: torch.exp(-(x**2)),
        ]

    def apply_gradient_checkpointing(self):
        """
        Apply gradient checkpointing to save memory during training.
        """
        try:
            self.basis_function = torch.utils.checkpoint(self.basis_function)
            self.layer_norm = torch.utils.checkpoint(self.layer_norm)
            self.batch_norm = torch.utils.checkpoint(self.batch_norm)
            self.adanorm = torch.utils.checkpoint(self.adanorm)
            logger.info("Gradient checkpointing applied successfully.")
        except Exception as e:
            logger.warning(
                f"Error applying gradient checkpointing: {e}. Continuing without checkpointing."
            )

    def apply_mixed_precision(self):
        """
        Apply mixed precision training to optimize computation.
        """
        try:
            self.basis_function = self.basis_function.half()
            self.layer_norm = self.layer_norm.half()
            self.batch_norm = self.batch_norm.half()
            self.adanorm = self.adanorm.half()
            self.param = self.param.half() if self.param is not None else None
            self.activation_weights = self.activation_weights.half()
            self.neuron_type_weights = self.neuron_type_weights.half()
            logger.info("Mixed precision applied successfully.")
        except Exception as e:
            logger.warning(
                f"Error applying mixed precision: {e}. Continuing without mixed precision."
            )

    @staticmethod
    def lstm_neuron(x: torch.Tensor) -> torch.Tensor:
        """
        Apply LSTM neuron processing to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after LSTM processing.
        """
        print("Applying LSTM neuron processing.")
        logger.info("Applying LSTM neuron processing.")
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if x.dim() != 3:
            raise ValueError(
                "Input tensor must have 3 dimensions (batch_size, seq_len, feature_size)"
            )

        batch_size, seq_len, feature_size = x.size()
        print(f"Input tensor size: {batch_size, seq_len, feature_size}")
        logger.info(f"Input tensor size: {batch_size, seq_len, feature_size}")
        h0 = torch.zeros(1, batch_size, feature_size, device=x.device)
        c0 = torch.zeros(1, batch_size, feature_size, device=x.device)
        print(f"Initial hidden state size: {h0.size()}")
        logger.info(f"Initial hidden state size: {h0.size()}")
        print(f"Initial cell state size: {c0.size()}")
        logger.info(f"Initial cell state size: {c0.size()}")
        lstm = nn.LSTM(
            input_size=feature_size, hidden_size=feature_size, batch_first=True
        ).to(x.device)
        print(
            f"LSTM module created with input size {feature_size} and hidden size {feature_size}"
        )
        logger.info(
            f"LSTM module created with input size {feature_size} and hidden size {feature_size}"
        )

        try:
            out, _ = lstm(x, (h0, c0))
            print(f"LSTM output size: {out.size()}")
            logger.info(f"LSTM output size: {out.size()}")
            print("LSTM neuron processing completed.")
            logger.info("LSTM neuron processing completed.")
            return out[:, -1, :].clamp(min=-1e6, max=1e6)
        except Exception as e:
            logger.warning(f"Error in LSTM neuron: {e}. Returning default output.")
            print(f"Error in LSTM neuron: {e}. Returning default output.")
            return x.clamp(min=-1e6, max=1e6)

    @staticmethod
    def gru_neuron(x: torch.Tensor) -> torch.Tensor:
        """
        Apply GRU neuron processing to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after GRU processing.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if x.dim() != 3:
            raise ValueError(
                "Input tensor must have 3 dimensions (batch_size, seq_len, feature_size)"
            )

        batch_size, seq_len, feature_size = x.size()
        h0 = torch.zeros(1, batch_size, feature_size, device=x.device)
        gru = nn.GRU(
            input_size=feature_size, hidden_size=feature_size, batch_first=True
        ).to(x.device)

        try:
            out, _ = gru(x, h0)
            return out[:, -1, :].clamp(min=-1e6, max=1e6)
        except Exception as e:
            logger.warning(f"Error in GRU neuron: {e}. Returning default output.")
            print(f"Error in GRU neuron: {e}. Returning default output.")
            return x.clamp(min=-1e6, max=1e6)

    @staticmethod
    def attention_neuron(x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention neuron processing to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after attention processing.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if x.dim() != 3:
            raise ValueError(
                "Input tensor must have 3 dimensions (batch_size, seq_len, feature_size)"
            )

        batch_size, seq_len, feature_size = x.size()
        attention = nn.MultiheadAttention(
            embed_dim=feature_size, num_heads=1, batch_first=True
        ).to(x.device)

        try:
            attn_output, _ = attention(x, x, x)
            return attn_output.mean(dim=1).clamp(min=-1e6, max=1e6)
        except Exception as e:
            logger.warning(f"Error in attention neuron: {e}. Returning default output.")
            print(f"Error in attention neuron: {e}. Returning default output.")
            return x.clamp(min=-1e6, max=1e6)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["activation_functions"] = [f.__name__ for f in self.activation_functions]
        state["neuron_types"] = [f.__name__ for f in self.neuron_types]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.activation_functions = [
            getattr(F, name) if hasattr(F, name) else getattr(self, name)
            for name in state["activation_functions"]
        ]
        self.neuron_types = [getattr(self, name) for name in state["neuron_types"]]

    def save(self, path: str) -> None:
        try:
            torch.save(self.state_dict(), path)
            logger.info(f"Model state saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model state: {e}")
            raise

    def load(self, path: str) -> None:
        try:
            self.load_state_dict(torch.load(path))
            logger.info(f"Model state loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model state: {e}")
            raise

    def log_activations(self, x: torch.Tensor) -> None:
        """
        Log the activations for debugging purposes.

        Parameters:
        x (torch.Tensor): Input tensor.
        """
        logger.debug(f"Activations: {x}")

    def train_on_synthetic_data(self):
        """
        Train the model on synthetic data.
        """
        try:
            X, y = make_regression(
                n_samples=SYNTHETIC_DATA_SIZE,
                n_features=self.input_size,
                noise=0.1,
                random_state=RANDOM_STATE,
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )

            train_dataset = TensorDataset(
                torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
            )
            test_dataset = TensorDataset(
                torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()
            )
            train_loader = DataLoader(
                train_dataset, batch_size=BATCH_SIZE, shuffle=True
            )
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

            optimizer = torch.optim.Adam(
                self.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
            )
            criterion = nn.MSELoss()

            train_losses, test_losses, test_rmses, test_mapes = [], [], [], []
            best_test_loss = float("inf")
            epochs_without_improvement = 0

            for epoch in range(MAX_EPOCHS):
                self.train()
                train_loss = 0.0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self(batch_X)
                    loss = criterion(outputs, batch_y.unsqueeze(1))
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * batch_X.size(0)
                train_loss /= len(train_loader.dataset)
                train_losses.append(train_loss)

                self.eval()
                test_loss = 0.0
                test_rmse = 0.0
                test_mape = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        outputs = self(batch_X)
                        loss = criterion(outputs, batch_y.unsqueeze(1))
                        test_loss += loss.item() * batch_X.size(0)
                        test_rmse += torch.sqrt(
                            nn.functional.mse_loss(outputs, batch_y.unsqueeze(1))
                        ) * batch_X.size(0)
                        test_mape += (
                            torch.mean(
                                torch.abs(
                                    (outputs - batch_y.unsqueeze(1))
                                    / batch_y.unsqueeze(1)
                                )
                            )
                            * 100
                            * batch_X.size(0)
                        )
                test_loss /= len(test_loader.dataset)
                test_rmse /= len(test_loader.dataset)
                test_mape /= len(test_loader.dataset)
                test_losses.append(test_loss)
                test_rmses.append(test_rmse)
                test_mapes.append(test_mape)

                logger.info(
                    f"Epoch {epoch+1}/{MAX_EPOCHS}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test RMSE: {test_rmse:.4f}, Test MAPE: {test_mape:.2f}%"
                )
                print(
                    f"Epoch {epoch+1}/{MAX_EPOCHS}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test RMSE: {test_rmse:.4f}, Test MAPE: {test_mape:.2f}%"
                )

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= 10:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.plot(train_losses, label="Train Loss")
            plt.plot(test_losses, label="Test Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()

            plt.subplot(2, 2, 2)
            plt.plot(test_rmses, label="Test RMSE")
            plt.xlabel("Epoch")
            plt.ylabel("RMSE")
            plt.legend()

            plt.subplot(2, 2, 3)
            plt.plot(test_mapes, label="Test MAPE")
            plt.xlabel("Epoch")
            plt.ylabel("MAPE (%)")
            plt.legend()

            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.error(f"Error during synthetic data training: {e}")
            print(f"Error during synthetic data training: {e}")
            raise

    def validate_output(self, output: torch.Tensor) -> None:
        """
        Validate the output tensor.

        Args:
            output (torch.Tensor): Output tensor to validate.

        Raises:
            TypeError: If output is not a torch.Tensor.
            ValueError: If output tensor dimensions or size do not match expectations.
        """
        print("Validating output tensor.")
        logger.info("Validating output tensor.")
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Output must be a torch.Tensor. Got {type(output)}")
        if output.dim() <= 1:
            raise ValueError(
                f"Output tensor must have more than 1 dimensions. Got {output.dim()}"
            )
        if output.size(-1) != self.input_size:
            raise ValueError(
                f"Output tensor size must match input size. Got {output.size(-1)} instead of {self.input_size}"
            )
        print("Output tensor validation passed.")
        logger.info("Output tensor validation passed.")

    def test(self):
        """
        Test the DynamicActivationNeuron with a sample input.
        """
        print("Starting test.")
        logger.info("Starting test.")
        try:
            input_size = self.input_size
            sample_input = torch.randn(2, input_size)
            print(f"Sample input: {sample_input}")
            logger.info(f"Sample input: {sample_input}")
            output = self.forward(sample_input)
            print(f"Output: {output}")
            logger.info(f"Output: {output}")
            self.validate_output(output)
            logger.info("Test completed successfully.")
            print("Test completed successfully.")
        except Exception as e:
            logger.error(f"Error during test: {e}")
            print(f"Error during test: {e}")
            raise


if __name__ == "__main__":
    try:
        input_size = 100
        dan = DynamicActivationNeuron(input_size)
        dan.test()
        dan.train_on_synthetic_data()
        dan.save("dan_model.pth")
        dan.load("dan_model.pth")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"Error in main execution: {e}")
        raise
