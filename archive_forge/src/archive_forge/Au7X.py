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
import json


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


def save_model(model: nn.Module, file_path: str, format: str = "pkl") -> None:
    if format == "pkl":
        torch.save(model, file_path)
    elif format == "json":
        with open(file_path, "w") as f:
            json.dump(model.state_dict(), f)
    elif format == "txt":
        with open(file_path, "w") as f:
            f.write(str(model.state_dict()))
    else:
        raise ValueError(f"Unsupported format: {format}")
    logger.info(f"Model saved to {file_path} in {format} format.")


def load_model(file_path: str, format: str = "pkl") -> nn.Module:
    model = DynamicActivationNeuron(INPUT_SIZE)
    if format == "pkl":
        model = torch.load(file_path)
    elif format == "json":
        with open(file_path, "r") as f:
            state_dict = json.load(f)
        model.load_state_dict(state_dict)
    elif format == "txt":
        with open(file_path, "r") as f:
            state_dict = eval(f.read())
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unsupported format: {format}")
    logger.info(f"Model loaded from {file_path} in {format} format.")
    return model


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
        self.param = (
            nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            if scale_output
            else None
        )

        # Define the default activation functions based on the Kolmogorov-Arnold representation theorem
        self.activation_functions = (
            activation_functions or self.default_activation_functions()
        )

        # Define the default neuron types based on the Kolmogorov-Arnold representation theorem
        self.neuron_types = neuron_types or self.default_neuron_types()

        # Define learnable weights for the activation functions and neuron types
        self.activation_weights = nn.Parameter(
            torch.randn(len(self.activation_functions), dtype=torch.float32)
        )
        self.neuron_type_weights = nn.Parameter(
            torch.randn(len(self.neuron_types), dtype=torch.float32)
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
            return torch.tensor(x, dtype=torch.float32)
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
        h0 = torch.zeros(
            1, batch_size, feature_size, device=x.device, dtype=torch.float32
        )
        c0 = torch.zeros(
            1, batch_size, feature_size, device=x.device, dtype=torch.float32
        )
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
        h0 = torch.zeros(
            1, batch_size, feature_size, device=x.device, dtype=torch.float32
        )
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


class DirectoryManager:
    @staticmethod
    def ensure_directories_exist(paths: List[str]) -> None:
        """
        Ensure all necessary directories exist.

        Args:
            paths (List[str]): A list of directory paths to ensure existence.
        """
        for path in paths:
            os.makedirs(path, exist_ok=True)
            logger.info(f"Ensured directory exists: {path}")


class DataProcessor:
    @staticmethod
    def parse_md_files(directory: str) -> List[str]:
        """Parse markdown files in a directory and return their contents."""
        texts = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".md"):
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        texts.append(f.read())
        return texts

    @staticmethod
    def extract_labels(texts: List[str]) -> List[int]:
        """Extract labels from markdown texts."""
        labels = []
        for text in texts:
            if "## USER" in text:
                labels.append(0)
            elif "## ASSISTANT" in text:
                labels.append(1)
            else:
                labels.append(-1)  # Unknown label
        return labels

    @staticmethod
    def process_and_prepare_md_data(
        md_directory: str,
        vectorizer: TfidfVectorizer,
        input_size: int,
        test_size: float,
        random_state: int,
        batch_size: int,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[DataLoader, DataLoader, int, int]:
        """
        Process and prepare markdown data for training and validation.
        """
        texts = DataProcessor.parse_md_files(md_directory)
        labels = DataProcessor.extract_labels(texts)

        # Filter out texts with unknown labels
        texts, labels = zip(
            *[(text, label) for text, label in zip(texts, labels) if label != -1]
        )

        # Ensure the vectorizer is set up with the correct input size
        vectorizer.max_features = input_size
        X = vectorizer.fit_transform(texts).toarray()

        # Check if the number of features matches the input size
        if X.shape[1] != input_size:
            raise ValueError(
                f"Feature size of the vectorized texts {X.shape[1]} does not match the expected input size {input_size}."
            )

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, labels, test_size=test_size, random_state=random_state
        )

        # Dynamically determine the output size
        unique_labels = np.unique(labels)
        output_size = len(unique_labels)

        # Convert to tensors with specified dtype
        X_train_tensor = torch.tensor(X_train, dtype=dtype)
        X_val_tensor = torch.tensor(X_val, dtype=dtype)
        y_train_tensor = torch.tensor(y_train, dtype=dtype)
        y_val_tensor = torch.tensor(y_val, dtype=dtype)

        # Create DataLoader for training and validation sets
        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(X_val_tensor, y_val_tensor),
            batch_size=batch_size,
            shuffle=False,
        )

        return train_loader, val_loader, input_size, output_size

    @staticmethod
    def create_or_load_synthetic_data(
        input_size: int,
        n_samples: int,
        test_size: float,
        random_state: int,
        synthetic_data_dir: str,
        batch_size: int,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create synthetic dataset if it doesn't exist, otherwise load it.
        """
        synthetic_data_path = os.path.join(synthetic_data_dir, "synthetic_data.npy")
        synthetic_labels_path = os.path.join(synthetic_data_dir, "synthetic_labels.npy")

        if not os.path.exists(synthetic_data_path):
            X, y = make_regression(
                n_samples=n_samples,
                n_features=input_size,
                noise=0.1,
            )
            np.save(synthetic_data_path, X)
            np.save(synthetic_labels_path, y)
            logger.info("Created and saved synthetic dataset.")
        else:
            X = np.load(synthetic_data_path)
            y = np.load(synthetic_labels_path)
            logger.info("Loaded existing synthetic dataset.")

        # Dynamically determine the output size
        output_size = y.shape[1] if y.ndim > 1 else 1

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Convert to tensors with specified dtype
        X_train_tensor = torch.tensor(X_train, dtype=dtype)
        X_val_tensor = torch.tensor(X_val, dtype=dtype)
        y_train_tensor = torch.tensor(y_train, dtype=dtype)
        y_val_tensor = torch.tensor(y_val, dtype=dtype)

        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(X_val_tensor, y_val_tensor),
            batch_size=batch_size,
            shuffle=False,
        )

        return train_loader, val_loader

    @staticmethod
    def create_new_regression_dataset(
        input_size: int,
        n_samples: int,
        test_size: float,
        random_state: int,
        synthetic_data_dir: str,
        batch_size: int,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create a new regression dataset and save it.
        """
        X, y = make_regression(
            n_samples=n_samples,
            n_features=input_size,
            noise=0.1,
        )

        os.makedirs(synthetic_data_dir, exist_ok=True)
        np.save(os.path.join(synthetic_data_dir, "new_synthetic_data.npy"), X)
        np.save(os.path.join(synthetic_data_dir, "new_synthetic_labels.npy"), y)
        logger.info("Created and saved new synthetic regression dataset.")

        # Dynamically determine the output size
        output_size = y.shape[1] if y.ndim > 1 else 1

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Convert to tensors with specified dtype
        X_train_tensor = torch.tensor(X_train, dtype=dtype)
        X_val_tensor = torch.tensor(X_val, dtype=dtype)
        y_train_tensor = torch.tensor(y_train, dtype=dtype)
        y_val_tensor = torch.tensor(y_val, dtype=dtype)

        # Create DataLoader for training and validation sets
        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(X_val_tensor, y_val_tensor),
            batch_size=batch_size,
            shuffle=False,
        )

        return train_loader, val_loader

    @staticmethod
    def create_new_text_dataset(
        md_directory: str,
        vectorizer: TfidfVectorizer,
        input_size: int,
        test_size: float,
        random_state: int,
        batch_size: int,
        processed_data_dir: str,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create a new text dataset from markdown files and save it.
        """
        texts = DataProcessor.parse_md_files(md_directory)
        labels = DataProcessor.extract_labels(texts)

        # Filter out texts with unknown labels
        texts, labels = zip(
            *[(text, label) for text, label in zip(texts, labels) if label != -1]
        )

        vectorizer.max_features = input_size
        X = vectorizer.fit_transform(texts).toarray()

        if X.shape[1] != input_size:
            raise ValueError(
                f"Feature size of the vectorized texts {X.shape[1]} does not match the expected input size {input_size}."
            )

        os.makedirs(processed_data_dir, exist_ok=True)
        np.savetxt(
            os.path.join(processed_data_dir, "new_combined_dataset.txt"), X, fmt="%f"
        )
        logger.info("Created and saved new text dataset.")

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, labels, test_size=test_size, random_state=random_state
        )

        # Dynamically determine the output size
        unique_labels = np.unique(labels)
        output_size = len(unique_labels)

        # Convert to tensors with specified dtype
        X_train_tensor = torch.tensor(X_train, dtype=dtype)
        X_val_tensor = torch.tensor(X_val, dtype=dtype)
        y_train_tensor = torch.tensor(y_train, dtype=dtype)
        y_val_tensor = torch.tensor(y_val, dtype=dtype)

        # Create DataLoader for training and validation sets
        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(X_val_tensor, y_val_tensor),
            batch_size=batch_size,
            shuffle=False,
        )

        return train_loader, val_loader


class ModelTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        path: str,
        target_loss: float,
        target_rmse: float,
        target_mape: float,
        max_epochs: int = 1000,
        save_interval: int = 10,
        overfit_thresholds: Optional[List[Tuple[float, int]]] = None,
        noise_std: float = 0.01,
        lr_decay: float = 0.9,
        weight_decay_increase: float = 1.1,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.path = path
        self.target_loss = target_loss
        self.target_rmse = target_rmse
        self.target_mape = target_mape
        self.max_epochs = max_epochs
        self.save_interval = save_interval
        self.overfit_thresholds = overfit_thresholds or [
            (100, 2),
            (50, 4),
            (25, 6),
            (20, 8),
            (10, 10),
        ]
        self.noise_std = noise_std
        self.lr_decay = lr_decay
        self.weight_decay_increase = weight_decay_increase
        self.history = self._initialize_history()
        self.epoch = 0
        self.overfit_count = 0
        self.initial_lr = optimizer.param_groups[0]["lr"]
        self.initial_weight_decay = optimizer.param_groups[0]["weight_decay"]
        self.stop_training = False

    def _initialize_history(self) -> Dict[str, List[float]]:
        return {
            "train_loss": [],
            "val_loss": [],
            "train_rmse": [],
            "val_rmse": [],
            "train_mape": [],
            "val_mape": [],
        }

    def calculate_metrics(
        self, outputs: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, float, float]:
        loss = self.criterion(outputs, labels)
        rmse = torch.sqrt(torch.mean((outputs - labels) ** 2)).item()
        mape = torch.mean(torch.abs((labels - outputs) / (labels + 1e-8))).item()
        return loss, rmse, mape

    def get_overfit_threshold(self, val_loss: float) -> int:
        for threshold, epochs in self.overfit_thresholds:
            if val_loss > threshold:
                return epochs
        return 10

    def user_wants_to_stop(self) -> bool:
        stop_signal_received = False

        def signal_handler(sig, frame):
            nonlocal stop_signal_received
            stop_signal_received = True

        prev_handler = signal.signal(signal.SIGUSR1, signal_handler)

        if stop_signal_received:
            print("Stop signal received. Finalizing training...")
            signal.signal(signal.SIGUSR1, prev_handler)
            return True
        else:
            signal.signal(signal.SIGUSR1, prev_handler)
            return False

    def train_and_save_model(self) -> Dict[str, List[float]]:
        while self.epoch < self.max_epochs and not self.stop_training:
            self._train_one_epoch()
            self._validate_one_epoch()

            logger.info(
                f"Epoch {self.epoch + 1}, Train Loss: {self.history['train_loss'][-1]:.4f}, "
                f"Val Loss: {self.history['val_loss'][-1]:.4f}, Train RMSE: {self.history['train_rmse'][-1]:.4f}, "
                f"Val RMSE: {self.history['val_rmse'][-1]:.4f}, Train MAPE: {self.history['train_mape'][-1]:.4f}, "
                f"Val MAPE: {self.history['val_mape'][-1]:.4f}"
            )

            if (self.epoch + 1) % self.save_interval == 0:
                self._save_model()
                Plotter.plot_metrics(self.history, self.path)
                Plotter.clear_memory()

            if self._check_target_metrics():
                break

            self._handle_overfitting()

            self.epoch += 1

            if self.user_wants_to_stop():
                logger.info("Training stopped by user.")
                self.stop_training = True

        if not self.stop_training:
            self._save_model()
        return self.history

    def _train_one_epoch(self) -> None:
        self.model.train()
        running_train_loss, running_train_rmse, running_train_mape = 0.0, 0.0, 0.0

        for inputs, labels in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss, rmse, mape = self.calculate_metrics(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            running_train_rmse += rmse * inputs.size(0)
            running_train_mape += mape * inputs.size(0)

        self._update_history(
            "train", running_train_loss, running_train_rmse, running_train_mape
        )

    def _validate_one_epoch(self) -> None:
        self.model.eval()
        running_val_loss, running_val_rmse, running_val_mape = 0.0, 0.0, 0.0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                outputs = self.model(inputs)
                loss, rmse, mape = self.calculate_metrics(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                running_val_rmse += rmse * inputs.size(0)
                running_val_mape += mape * inputs.size(0)

        self._update_history(
            "val", running_val_loss, running_val_rmse, running_val_mape
        )

    def _update_history(
        self, phase: str, loss: float, rmse: float, mape: float
    ) -> None:
        dataset_size = (
            len(self.train_loader.dataset)
            if phase == "train"
            else len(self.val_loader.dataset)
        )
        self.history[f"{phase}_loss"].append(loss / dataset_size)
        self.history[f"{phase}_rmse"].append(rmse / dataset_size)
        self.history[f"{phase}_mape"].append(mape / dataset_size)

    def _save_model(self) -> None:
        torch.save(self.model.state_dict(), self.path)
        logger.info(f"Saved the model state to {self.path} at epoch {self.epoch + 1}.")

    def _check_target_metrics(self) -> bool:
        if (
            self.history["val_loss"][-1] <= self.target_loss
            or self.history["val_rmse"][-1] <= self.target_rmse
            or self.history["val_mape"][-1] <= self.target_mape
        ):
            logger.info(
                f"Target metric reached at epoch {self.epoch + 1}. "
                f"Loss: {self.history['val_loss'][-1]:.4f}, RMSE: {self.history['val_rmse'][-1]:.4f}, "
                f"MAPE: {self.history['val_mape'][-1]:.4f}"
            )
            return True
        return False

    def _handle_overfitting(self) -> None:
        if self.epoch > 0:
            current_val_loss = self.history["val_loss"][-1]
            previous_val_loss = self.history["val_loss"][-2]
            overfit_epochs_required = self.get_overfit_threshold(current_val_loss)

            if current_val_loss > previous_val_loss:
                self.overfit_count += 1
            else:
                self.overfit_count = 0

            if self.overfit_count >= overfit_epochs_required:
                logger.info("Overfitting detected, adding noise to parameters.")
                self._add_noise_to_parameters()
                self._adjust_optimizer_params()
                self.overfit_count = 0

    def _add_noise_to_parameters(self) -> None:
        with torch.no_grad():
            for param in self.model.parameters():
                noise = torch.normal(mean=0.0, std=self.noise_std, size=param.size())
                param.add_(noise)

    def _adjust_optimizer_params(self) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = max(
                param_group["lr"] * self.lr_decay, self.initial_lr * 0.1
            )
            param_group["weight_decay"] = min(
                param_group["weight_decay"] * self.weight_decay_increase,
                self.initial_weight_decay * 10,
            )


class Predictor:
    @staticmethod
    def predict(model: nn.Module, data_loader: DataLoader) -> torch.Tensor:
        """
        Generate predictions using the provided model and data loader.

        Args:
            model (nn.Module): The model to use for predictions.
            data_loader (DataLoader): The data loader providing input data.

        Returns:
            torch.Tensor: Concatenated predictions from the model.
        """
        model.eval()
        predictions = []
        with torch.no_grad():
            for inputs, _ in data_loader:
                outputs = model(inputs)
                predictions.append(outputs)
        return torch.cat(predictions, dim=0)

    @staticmethod
    def log_predictions(predictions: torch.Tensor) -> None:
        """
        Log the mean and standard deviation of the predictions.

        Args:
            predictions (torch.Tensor): The predictions to analyze.
        """
        try:
            mean_prediction = torch.mean(predictions, dim=0)
            std_prediction = torch.std(predictions, dim=0)
            logger.info(f"Mean of predictions: {mean_prediction}")
            logger.info(f"Standard deviation of predictions: {std_prediction}")
        except Exception as e:
            logger.error(f"Error during prediction analysis: {e}")

    @staticmethod
    def save_predictions(predictions: torch.Tensor, file_path: str) -> None:
        """
        Save the predictions to a specified file.

        Args:
            predictions (torch.Tensor): The predictions to save.
            file_path (str): The path to the file where predictions will be saved.
        """
        try:
            np.savetxt(file_path, predictions.cpu().numpy(), fmt="%f")
            logger.info(f"Predictions saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")

    @staticmethod
    def load_predictions(file_path: str) -> torch.Tensor:
        """
        Load predictions from a specified file.

        Args:
            file_path (str): The path to the file from which predictions will be loaded.

        Returns:
            torch.Tensor: The loaded predictions.
        """
        try:
            predictions = torch.tensor(np.loadtxt(file_path), dtype=torch.float32)
            logger.info(f"Predictions loaded from {file_path}")
            return predictions
        except Exception as e:
            logger.error(f"Error loading predictions: {e}")
            return torch.tensor([])

    @staticmethod
    def analyze_predictions(predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze the predictions to compute mean and standard deviation.

        Args:
            predictions (torch.Tensor): The predictions to analyze.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing mean and standard deviation of predictions.
        """
        return {
            "mean": torch.mean(predictions, dim=0),
            "std": torch.std(predictions, dim=0),
        }

    @staticmethod
    def save_analysis(analysis: Dict[str, torch.Tensor], file_path: str) -> None:
        """
        Save the analysis of predictions to a specified file.

        Args:
            analysis (Dict[str, torch.Tensor]): The analysis to save.
            file_path (str): The path to the file where analysis will be saved.
        """
        try:
            with open(file_path, "w") as f:
                for key, value in analysis.items():
                    f.write(f"{key}: {value.cpu().numpy()}\n")
            logger.info(f"Analysis saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")


class Plotter:
    @staticmethod
    def plot_metrics(history: Dict[str, List[float]], output_dir: str) -> None:
        """
        Plot training and validation metrics over epochs and save the plots.

        Args:
            history (Dict[str, List[float]]): Dictionary containing training and validation metrics.
            output_dir (str): Directory to save the plots.
        """
        epochs = range(len(history["train_loss"]))
        plt.figure(figsize=(18, 10))

        metrics = {
            "loss": ("Loss", "Training and Validation Loss Over Time"),
            "rmse": ("RMSE", "Training and Validation RMSE Over Time"),
            "mape": ("MAPE", "Training and Validation MAPE Over Time"),
        }

        for i, (metric, (ylabel, title)) in enumerate(metrics.items(), start=1):
            Plotter._plot_metric(epochs, history, metric, ylabel, title, i)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"metrics_epoch_{len(epochs)}.png"))
        plt.show(block=False)
        plt.pause(5)
        plt.close()

    @staticmethod
    def _plot_metric(
        epochs: range,
        history: Dict[str, List[float]],
        metric: str,
        ylabel: str,
        title: str,
        subplot_index: int,
    ) -> None:
        """
        Plot a specific metric.

        Args:
            epochs (range): Range of epochs.
            history (Dict[str, List[float]]): Dictionary containing training and validation metrics.
            metric (str): Metric to plot.
            ylabel (str): Label for the y-axis.
            title (str): Title of the plot.
            subplot_index (int): Index of the subplot.
        """
        plt.subplot(2, 2, subplot_index)
        plt.plot(epochs, history[f"train_{metric}"], label=f"Training {ylabel}")
        plt.plot(epochs, history[f"val_{metric}"], label=f"Validation {ylabel}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{title} (Best Val {ylabel}: {min(history[f'val_{metric}']):.4f})")
        plt.legend()

    @staticmethod
    def clear_memory() -> None:
        """
        Clear GPU memory and perform garbage collection.
        """
        torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def get_user_choice(prompt: str) -> bool:
        """
        Prompt the user for a yes/no choice.

        Args:
            prompt (str): The prompt message to display.

        Returns:
            bool: True if the user chooses 'Y', False otherwise.
        """
        while True:
            choice = input(f"{prompt} (Y/N): ").strip().upper()
            if choice in ["Y", "N"]:
                return choice == "Y"
            print("Invalid input. Please enter 'Y' or 'N'.")


def main() -> None:
    """
    Main function to test and train the DynamicActivationNeuron module.
    """

    # Ensure all necessary directories exist
    paths = [SYNTHETIC_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, ROOT_DIR]
    DirectoryManager.ensure_directories_exist(paths)

    # Load or create a model
    from tkinter import Tk, filedialog

    # Hide the root window
    root = Tk()
    root.withdraw()

    # Open file dialog
    model_path = filedialog.askopenfilename(
        initialdir=os.path.dirname(ROOT_DIR),
        title="Select a Model File",
        filetypes=(("Pickle files", "*.pkl"), ("All files", "*.*")),
    )

    if model_path:
        dan = torch.load(model_path)
        logger.info(f"Loaded model from {model_path}.")
    else:
        dan = DynamicActivationNeuron(INPUT_SIZE)
        torch.save(dan, UNTRAINED_MODEL_PATH)
        logger.info("Created and saved a new untrained model.")

    # Train the model with text data
    def train_with_text_data(dan: DynamicActivationNeuron) -> None:
        """Train the model with text data."""
        if use_existing_data:
            text_train_loader, text_val_loader, _, _ = (
                DataProcessor.process_and_prepare_md_data(
                    MD_DIRECTORY,
                    VECTORIZER,
                    INPUT_SIZE,
                    TEST_SIZE,
                    RANDOM_STATE,
                    BATCH_SIZE,
                )
            )
        else:
            text_train_loader, text_val_loader = DataProcessor.create_new_text_dataset(
                MD_DIRECTORY,
                VECTORIZER,
                INPUT_SIZE,
                TEST_SIZE,
                RANDOM_STATE,
                BATCH_SIZE,
                PROCESSED_DATA_DIR,
            )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(dan.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        model_handler = ModelTrainer(
            dan,
            criterion,
            optimizer,
            text_train_loader,
            text_val_loader,
            TEXT_TRAINED_MODEL_PATH,
            TARGET_LOSS,
            TARGET_RMSE,
            TARGET_MAPE,
        )
        history = model_handler.train_and_save_model()
        Plotter.plot_metrics(history, OUTPUT_DIR)
        Plotter.clear_memory()

        with open(TEXT_TRAINED_MODEL_PATH, "rb") as f:
            trained_model = pickle.load(f)

        predictions = Predictor.predict(trained_model, text_val_loader)
        Predictor.log_predictions(predictions)

    # Train the model with synthetic data
    def train_with_synthetic_data(dan: DynamicActivationNeuron) -> None:
        """Train the model with synthetic data."""
        if use_existing_data:
            regression_train_loader, regression_val_loader = (
                DataProcessor.create_or_load_synthetic_data(
                    INPUT_SIZE,
                    SYNTHETIC_DATA_SIZE,
                    TEST_SIZE,
                    RANDOM_STATE,
                    SYNTHETIC_DATA_DIR,
                    BATCH_SIZE,
                )
            )
        else:
            regression_train_loader, regression_val_loader = (
                DataProcessor.create_new_regression_dataset(
                    INPUT_SIZE,
                    SYNTHETIC_DATA_SIZE,
                    TEST_SIZE,
                    RANDOM_STATE,
                    SYNTHETIC_DATA_DIR,
                    BATCH_SIZE,
                )
            )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(dan.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        model_handler = ModelTrainer(
            dan,
            criterion,
            optimizer,
            regression_train_loader,
            regression_val_loader,
            REGRESSION_TRAINED_MODEL_PATH,
            TARGET_LOSS,
            TARGET_RMSE,
            TARGET_MAPE,
        )
        history = model_handler.train_and_save_model()
        Plotter.plot_metrics(history, OUTPUT_DIR)
        Plotter.clear_memory()

        with open(REGRESSION_TRAINED_MODEL_PATH, "rb") as f:
            trained_model = pickle.load(f)

        predictions = Predictor.predict(trained_model, regression_val_loader)
        Predictor.log_predictions(predictions)

    try:
        use_text_data = Plotter.get_user_choice("Do you want to train on text data?")
        use_existing_data = Plotter.get_user_choice(
            "Do you want to use existing dataset?"
        )

        if use_text_data:
            train_with_text_data(dan)
        else:
            train_with_synthetic_data(dan)

    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
