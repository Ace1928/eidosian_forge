import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
MD_DIRECTORY: str = "/home/lloyd/Downloads/gpt_chats"
PROCESSED_DATA_DIR: str = "/media/lloyd/Aurora_M2/dandata/processed_data/"
SYNTHETIC_DATA_DIR: str = "/media/lloyd/Aurora_M2/dandata/synthetic_data/"
UNTRAINED_MODEL_PATH: str = "/media/lloyd/Aurora_M2/dandata/untrained_dan_model.pkl"
REGRESSION_TRAINED_MODEL_PATH: str = (
    "/media/lloyd/Aurora_M2/dandata/regression_trained_dan_model.pkl"
)
TEXT_TRAINED_MODEL_PATH: str = (
    "/media/lloyd/Aurora_M2/dandata/text_trained_dan_model.pkl"
)
OUTPUT_DIR: str = "/media/lloyd/Aurora_M2/dandata/outputs/"
INPUT_SIZE: int = 1024
SYNTHETIC_DATA_SIZE: int = 1000000
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
LR: float = 0.001
WEIGHT_DECAY: float = 0.0001
SCALE_OUTPUT: bool = True
VECTORIZER: TfidfVectorizer = TfidfVectorizer(max_features=INPUT_SIZE)
BATCH_SIZE: int = 100
TARGET_LOSS: float = 0.01
MAX_EPOCHS: int = 1000


class DynamicActivationNeuron(nn.Module):
    def __init__(self, input_size):
        super(DynamicActivationNeuron, self).__init__()
        self.input_size = input_size
        self.activation_functions = []
        self.neuron_types = []
        self.activation_weights = nn.Parameter(
            torch.randn(len(self.activation_functions))
        )
        self.neuron_type_weights = nn.Parameter(torch.randn(len(self.neuron_types)))
        self.layer_norm = nn.LayerNorm(input_size)
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.adanorm = nn.Identity()  # Placeholder for adaptive normalization
        self.param = None  # Placeholder for any additional parameter
        self.basis_function = nn.Identity()  # Placeholder for basis function

    def _ensure_tensor(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        elif isinstance(x, list):
            return torch.tensor(x, dtype=torch.float)
        elif not isinstance(x, torch.Tensor):
            raise TypeError(
                f"Unsupported input type: {type(x)}. Expected numpy.ndarray, list, or torch.Tensor."
            )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._ensure_tensor(x).float()
        original_shape = x.shape
        print(f"Original shape: {original_shape}")

        x = x.view(-1, self.input_size)  # Flatten input to match input size
        print(f"Shape after flattening: {x.shape}")

        x = self.basis_function(x)
        print(f"Shape after basis function: {x.shape}")

        x = self.apply_normalization(x)
        print(f"Shape after normalization: {x.shape}")

        try:
            x = self.apply_learned_activation(x)
            print(f"Shape after learned activation: {x.shape}")

            x = self.apply_learned_neuron_type(x)
            print(f"Shape after learned neuron type: {x.shape}")
        except Exception as e:
            logger.error(f"Error in processing: {e}")
            raise RuntimeError(f"Processing failed: {e}")

        if self.param is not None:
            x = self.param * x
            print(f"Shape after applying param: {x.shape}")

        final_shape = list(original_shape[:-1]) + [x.shape[-1]]
        print(f"Final shape to be reshaped to: {final_shape}")
        x = x.view(*final_shape)
        print(f"Shape after final reshape: {x.shape}")

        return x.clamp(min=-1e6, max=1e6)

    def apply_learned_activation(self, x: torch.Tensor) -> torch.Tensor:
        if not self.activation_functions:
            return x
        try:
            activation_weights = F.softmax(self.activation_weights, dim=0)
            activation_output = sum(
                weight * activation(x)
                for weight, activation in zip(
                    activation_weights, self.activation_functions
                )
            )
            return activation_output.clamp(min=-1e6, max=1e6)
        except Exception as e:
            logger.error(f"Error in learned activation: {e}")
            raise RuntimeError(f"Activation processing failed: {e}")

    def apply_learned_neuron_type(self, x: torch.Tensor) -> torch.Tensor:
        if not self.neuron_types:
            return x
        try:
            neuron_type_weights = F.softmax(self.neuron_type_weights, dim=0)
            neuron_type_output = sum(
                weight * neuron(x)
                for weight, neuron in zip(neuron_type_weights, self.neuron_types)
            )
            return neuron_type_output.clamp(min=-1e6, max=1e6)
        except Exception as e:
            logger.error(f"Error in neuron type processing: {e}")
            raise RuntimeError(f"Neuron type processing failed: {e}")

    def validate_output(self, x: torch.Tensor) -> None:
        try:
            assert not torch.isnan(x).any(), "Output contains NaNs."
            assert not torch.isinf(x).any(), "Output contains Infs."
            assert (
                x.shape[-1] == self.output_size
            ), f"Output has incorrect shape. Expected last dimension size {self.output_size}, got {x.shape[-1]}"
        except AssertionError as e:
            logger.warning(f"Output validation failed: {e}")
            raise

    def test(self):
        try:
            input_size = self.input_size
            sample_input = torch.randn(2, input_size)
            output = self.forward(sample_input)
            self.validate_output(output)
            logger.info("Test completed successfully.")
        except Exception as e:
            logger.error(f"Error during test: {e}")
            raise

    def apply_normalization(self, x: torch.Tensor) -> torch.Tensor:
        try:
            x = self.layer_norm(x)
            if x.dim() == 2:
                x = self.batch_norm(x)
            elif x.dim() > 2:
                x_flat = x.view(-1, x.size(-1))
                x_norm = self.batch_norm(x_flat).view(x.size())
                x = x_norm
            else:
                logger.warning(
                    f"Unsupported tensor dimensionality: {x.dim()}. Skipping BatchNorm."
                )
            x = self.adanorm(x)
        except Exception as e:
            logger.warning(
                f"Error in normalization: {e}. Continuing without normalization."
            )
        return x

    @staticmethod
    def lstm_neuron(x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if x.dim() != 3:
            raise ValueError(
                "Input tensor must have 3 dimensions (batch_size, seq_len, feature_size)"
            )

        batch_size, seq_len, feature_size = x.size()
        h0 = torch.zeros(1, batch_size, feature_size, device=x.device)
        c0 = torch.zeros(1, batch_size, feature_size, device=x.device)
        lstm = nn.LSTM(
            input_size=feature_size, hidden_size=feature_size, batch_first=True
        ).to(x.device)

        try:
            out, _ = lstm(x, (h0, c0))
            return out[:, -1, :].clamp(min=-1e6, max=1e6)
        except Exception as e:
            logger.warning(f"Error in LSTM neuron: {e}. Returning default output.")
            return x.clamp(min=-1e6, max=1e6)

    @staticmethod
    def gru_neuron(x: torch.Tensor) -> torch.Tensor:
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
            return x.clamp(min=-1e6, max=1e6)

    @staticmethod
    def attention_neuron(x: torch.Tensor) -> torch.Tensor:
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
            return x.clamp(min=-1e6, max=1e6)

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

    def train_on_synthetic_data(self):
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

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= 10:
                    logger.info(f"Early stopping at epoch {epoch+1}")
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
            raise


if __name__ == "__main__":
    try:
        input_size = 10
        output_size = 1
        dan = DynamicActivationNeuron(input_size, output_size)
        dan.test()
        dan.train_on_synthetic_data()
        dan.save("dan_model.pth")
        dan.load("dan_model.pth")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise
