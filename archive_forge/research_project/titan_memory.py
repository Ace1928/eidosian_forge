import os
import shutil
import tempfile
import logging
import psutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Tuple
import unittest
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class MemoryConfig:
    input_dimension: int
    hidden_dimension: int
    forgetting_rate: float = 0.1
    surprise_learning_rate: float = 0.01
    surprise_momentum_decay: float = 0.9
    initial_chunk_size: int = 128
    min_chunk_size: int = 32
    max_chunk_size: int = 512
    offload_threshold_gb: float = 1.0
    num_threads: int = 4
    log_level: int = logging.INFO
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    use_deep_memory: bool = False  # Flag to use deep memory architecture
    adaptive_forgetting: bool = True # Flag to use adaptive forgetting
    deep_memory_layers: int = 2 # Number of layers in deep memory module

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeepTitanMemory(nn.Module):
    """
    Deep Memory module that extends the linear memory update with a multi-layer perceptron network.
    """
    def __init__(self, config: MemoryConfig):
        """
        Initializes the DeepTitanMemory module.

        Args:
            config (MemoryConfig): Configuration object containing hyperparameters.
        """
        super().__init__()
        self.config = config
        layers = []
        for i in range(config.deep_memory_layers):
            layers.append(nn.Linear(config.hidden_dimension, config.hidden_dimension))
            layers.append(nn.SiLU())
        self.mlp = nn.Sequential(*layers).to(config.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the deep memory module.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after passing through the MLP.
        """
        return self.mlp(x)


class TitanMemory(nn.Module):
    """
    TitanMemory module that incorporates SiLU activation, L2 normalization,
    chunk-wise processing, and disk offloading for efficient memory management.
    """
    offload_dir: Optional[str] = None

    def __init__(self, config: MemoryConfig):
        """
        Initializes the TitanMemory module.

        Args:
            config (MemoryConfig): Configuration object containing hyperparameters.
        """
        super().__init__()
        self.config = config
        logging.getLogger().setLevel(config.log_level)

        # Define linear projections for key, value, and query
        self.key_projection = nn.Linear(config.input_dimension, config.hidden_dimension, bias=False).to(config.device)
        self.value_projection = nn.Linear(config.input_dimension, config.hidden_dimension, bias=False).to(config.device)
        self.query_projection = nn.Linear(config.input_dimension, config.hidden_dimension, bias=False).to(config.device)

        # Register buffers for memory matrix and surprise term
        self.register_buffer(
            'memory_matrix',
            torch.zeros(config.hidden_dimension, config.hidden_dimension, device=config.device)
        )
        self.register_buffer(
            'surprise_term',
            torch.zeros(config.hidden_dimension, config.hidden_dimension, device=config.device)
        )

        # Setup thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=config.num_threads)

        # Initialize offload paths and flags
        self.memory_offload_path = None
        self.surprise_offload_path = None
        self.memory_offloaded = False
        self.surprise_offloaded = False
        self.chunk_offload_dir = None

        # Create temporary directory for offloading tensors if possible
        try:
            self.offload_dir = tempfile.mkdtemp()
            if self.offload_dir:
                self.memory_offload_path = os.path.join(self.offload_dir, "memory.pt")
                self.surprise_offload_path = os.path.join(self.offload_dir, "surprise.pt")
                self.chunk_offload_dir = os.path.join(self.offload_dir, "chunks")
                os.makedirs(self.chunk_offload_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Error creating temporary directory: {e}")
            self.offload_dir = None
            self.memory_offload_path = None
            self.surprise_offload_path = None
            self.chunk_offload_dir = None

        # Initialize deep memory module if enabled
        if self.config.use_deep_memory:
            self.deep_memory_update = DeepTitanMemory(config).to(config.device)
        else:
            self.deep_memory_update = None

        # Initialize adaptive forgetting parameters
        if self.config.adaptive_forgetting:
            self.forgetting_projection = nn.Linear(config.hidden_dimension, 1, bias=False).to(config.device)


    def cleanup(self):
        """
        Explicitly cleans up temporary directories and offloaded files.
        Should be called when the TitanMemory instance is no longer needed.
        """
        self._cleanup_offload_directory()

    def _cleanup_offload_directory(self):
        """
        Cleans up the offload directory and associated files.

        This method attempts to remove the temporary directory used for offloading,
        including any chunk subdirectories and individual tensor files. It logs
        warnings if the directory doesn't exist and errors if removal fails.
        """
        if not self.offload_dir:
            logger.warning("No offload directory to clean up.")
            return
        # Remove chunk directory
        if self.chunk_offload_dir and os.path.exists(self.chunk_offload_dir):
            try:
                shutil.rmtree(self.chunk_offload_dir)
                logger.debug(f"Removed chunk offload directory: {self.chunk_offload_dir}")
            except OSError as e:
                logger.error(f"Error removing chunk offload directory {self.chunk_offload_dir}: {e}")
        # Remove memory offload file
        if self.memory_offload_path and os.path.exists(self.memory_offload_path):
            try:
                os.remove(self.memory_offload_path)
                logger.debug(f"Removed memory offload file: {self.memory_offload_path}")
            except OSError as e:
                logger.error(f"Error removing memory offload file {self.memory_offload_path}: {e}")
        # Remove surprise offload file
        if self.surprise_offload_path and os.path.exists(self.surprise_offload_path):
            try:
                os.remove(self.surprise_offload_path)
                logger.debug(f"Removed surprise offload file: {self.surprise_offload_path}")
            except OSError as e:
                logger.error(f"Error removing surprise offload file {self.surprise_offload_path}: {e}")
        # Remove the main offload directory
        if self.offload_dir and os.path.exists(self.offload_dir):
            try:
                shutil.rmtree(self.offload_dir)
                logger.info(f"Removed temporary directory: {self.offload_dir}")
            except OSError as e:
                logger.error(f"Error removing temporary directory {self.offload_dir}: {e}")

    def _save_tensor(self, tensor: torch.Tensor, path: str) -> None:
        """Saves a tensor to disk at the specified path."""
        if not isinstance(tensor, torch.Tensor):
            logger.error(f"Expected torch.Tensor, but got {type(tensor)}")
            raise TypeError(f"Expected torch.Tensor, but got {type(tensor)}")
        try:
            torch.save(tensor.cpu(), path)
            logger.info(f"Offloaded tensor to {path}")
        except Exception as e:
            logger.error(f"Error saving tensor to {path}: {e}")

    def _load_tensor(self, path: str, device: torch.device) -> Optional[torch.Tensor]:
        """Loads a tensor from disk from the specified path onto the given device."""
        if not isinstance(device, torch.device):
            logger.error(f"Expected torch.device, but got {type(device)}")
            raise TypeError(f"Expected torch.device, but got {type(device)}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                tensor = torch.load(path, map_location=device)
            return tensor
        except FileNotFoundError:
            logger.error(f"File not found at {path}")
            return None
        except Exception as e:
            logger.error(f"Error loading tensor from {path}: {e}")
            return None

    def _offload_to_disk(self) -> None:
        """
        Offloads the memory matrix and surprise term to disk to free up GPU memory.

        This operation checks if the offload directory is initialized and if the
        respective tensors are currently in memory and not already offloaded.
        If these conditions are met, the tensors are saved to disk, and the
        in-memory references are set to None.
        """
        if not self.offload_dir:
            logger.warning("Offload directory not initialized, skipping offload.")
            return
        # Offload memory matrix if not already offloaded
        if not self.memory_offloaded and self.memory_matrix is not None and self.memory_offload_path:
            try:
                self._save_tensor(self.memory_matrix, self.memory_offload_path)
                self.memory_offloaded = True
                self.memory_matrix = None
                logger.info("Memory matrix offloaded to disk.")
            except Exception as e:
                logger.error(f"Error offloading memory matrix: {e}")
        # Offload surprise term if not already offloaded
        if not self.surprise_offloaded and self.surprise_term is not None and self.surprise_offload_path:
            try:
                self._save_tensor(self.surprise_term, self.surprise_offload_path)
                self.surprise_offloaded = True
                self.surprise_term = None
                logger.info("Surprise term offloaded to disk.")
            except Exception as e:
                logger.error(f"Error offloading surprise term: {e}")

    def _load_from_disk(self) -> None:
        """
        Loads the offloaded memory matrix and surprise term from disk if they were offloaded.

        This operation checks if the offload directory is available and if the
        respective tensors have been marked as offloaded. Upon successful loading,
        the tensor is moved to the configured device, and the offloaded flag is reset.
        """
        if not self.offload_dir:
            logger.warning("Offload directory not initialized, skipping load.")
            return
        # Load memory matrix if offloaded
        if self.memory_offloaded and self.memory_offload_path:
            loaded_tensor = self._load_tensor(self.memory_offload_path, self.config.device)
            if loaded_tensor is not None:
                self.memory_matrix = loaded_tensor.to(self.config.device)
                self.memory_offloaded = False
                logger.info("Memory matrix loaded from disk.")
        # Load surprise term if offloaded
        if self.surprise_offloaded and self.surprise_offload_path:
            loaded_tensor = self._load_tensor(self.surprise_offload_path, self.config.device)
            if loaded_tensor is not None:
                self.surprise_term = loaded_tensor.to(self.config.device)
                self.surprise_offloaded = False
                logger.info("Surprise term loaded from disk.")

    def _normalize_and_activate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies SiLU activation followed by L2 normalization on the given tensor.

        Args:
            x: Input tensor.

        Returns:
            Normalized and activated tensor.
        """
        if not isinstance(x, torch.Tensor):
            logger.error(f"Expected torch.Tensor, but got {type(x)}")
            raise TypeError("Input must be a torch.Tensor")
        activated = F.silu(x) if hasattr(F, 'silu') else (x * torch.sigmoid(x))
        return F.normalize(activated, p=2, dim=-1)

    def _compute_loss_and_gradient(
        self, prediction: torch.Tensor, value: torch.Tensor, key: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the squared error loss and its gradient with respect to the memory matrix.

        Args:
            prediction: Predicted value from memory.
            value: True value vector.
            key: Key vector used for current prediction.

        Returns:
            A tuple of (loss, gradient).
        """
        if not all(isinstance(t, torch.Tensor) for t in [prediction, value, key]):
            logger.error(f"Inputs must be torch.Tensors, got {[type(t) for t in [prediction, value, key]]}")
            raise TypeError("All inputs must be torch.Tensors")
        
        # Ensure prediction and value have the same shape
        if prediction.dim() > value.dim():
            prediction = prediction.squeeze(0)
        elif value.dim() > prediction.dim():
            value = value.unsqueeze(0)
        
        if prediction.shape != value.shape:
            logger.error(f"Prediction and value tensors must have the same shape. Got {prediction.shape} and {value.shape}")
            raise ValueError(f"Prediction and value tensors must have the same shape. Got {prediction.shape} and {value.shape}")
        
        if key.ndim != 1:
            logger.error(f"Key tensor must be 1-dimensional. Got {key.dim()} dimensions.")
            raise ValueError("Key tensor must be 1-dimensional")
        
        error = (prediction - value).unsqueeze(1)  # Shape: [batch_size, 1]
        loss = torch.norm(prediction - value)**2
        gradient = 2.0 * error @ key.unsqueeze(0)  # Shape: [batch_size, hidden_dim]
        return loss, gradient

    def _update_memory(self, gradient: torch.Tensor) -> None:
        """
        Updates the surprise term and memory matrix based on the given gradient.

        Args:
            gradient: Gradient of loss with respect to memory.
        """
        if not isinstance(gradient, torch.Tensor):
            logger.error(f"Expected torch.Tensor for gradient, but got {type(gradient)}")
            raise TypeError(f"Expected torch.Tensor for gradient, but got {type(gradient)}")
        if gradient.shape != self.memory_matrix.shape:
            logger.error(f"Gradient shape {gradient.shape} does not match memory matrix shape {self.memory_matrix.shape}")
            raise ValueError(f"Gradient shape {gradient.shape} must match memory matrix shape {self.memory_matrix.shape}")

        # Update surprise term with momentum
        self.surprise_term = (
            self.config.surprise_momentum_decay * self.surprise_term
            - self.config.surprise_learning_rate * gradient
        )

        # Compute adaptive forgetting rate
        alpha_t = self._compute_adaptive_forgetting_rate(gradient) if self.config.adaptive_forgetting else self.config.forgetting_rate

        # Update memory matrix with adaptive forgetting
        memory_update = self.deep_memory_update(self.surprise_term) if self.deep_memory_update is not None else self.surprise_term
        self.memory_matrix = (1 - alpha_t) * self.memory_matrix + memory_update

    def _compute_adaptive_forgetting_rate(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Computes the adaptive forgetting rate based on the gradient.

        Args:
            gradient: The gradient of the loss with respect to the memory matrix.

        Returns:
            The adaptive forgetting rate.
        """
        # Compute mean over the first dimension to get a vector of size (hidden_dim,)
        gradient_mean = gradient.mean(dim=0).unsqueeze(0)  # Shape: (1, hidden_dim)

        # Project the gradient to get the forgetting rate
        alpha_t = torch.sigmoid(self.forgetting_projection(gradient_mean))  # Shape: (1, 1)

        return alpha_t.clamp(0, 1)

    def _predict_and_update(self, current_input: torch.Tensor) -> torch.Tensor:
        """
        Predicts output using the current memory, updates the memory with the new input,
        and retrieves the result.

        Args:
            current_input: A single input token tensor.

        Returns:
            Output tensor for the current input.
        """
        if not isinstance(current_input, torch.Tensor):
            logger.error(f"Expected torch.Tensor, but got {type(current_input)}")
            raise TypeError("Input must be a torch.Tensor")

        # Load offloaded tensors if necessary
        if self.memory_offloaded:
            self._load_from_disk()

        # Project input to key and value vectors with activation and normalization
        key = self._normalize_and_activate(self.key_projection(current_input.to(self.config.device)))  # Shape: [hidden_dim]
        value = self._normalize_and_activate(self.value_projection(current_input.to(self.config.device)))  # Shape: [hidden_dim]

        # Ensure key is 1D
        if key.dim() != 1:
            key = key.squeeze(0)
            if key.dim() != 1:
                logger.error(f"Key tensor is not 1D after squeezing. Shape: {key.shape}")
                raise ValueError("Key tensor must be 1-dimensional after squeezing.")

        # Compute prediction from current memory
        prediction = self.memory_matrix @ key  # Shape: [hidden_dim]

        # Compute loss and gradient
        _, gradient = self._compute_loss_and_gradient(prediction, value, key)
        # Update memory based on computed gradient
        self._update_memory(gradient)

        # Project input to query vector with activation and normalization
        query = self._normalize_and_activate(self.query_projection(current_input.to(self.config.device)))
        # Retrieve output using updated memory and query
        return self.memory_matrix @ query

    def _process_chunk(self, chunk: torch.Tensor, chunk_index: int) -> torch.Tensor:
        """
        Processes a single chunk of the input sequence.

        Args:
            chunk: Tensor containing a chunk of the input sequence.
            chunk_index: Index of the current chunk.

        Returns:
            Output tensor for the processed chunk.
        """
        if not isinstance(chunk, torch.Tensor):
            logger.error(f"Expected torch.Tensor, but got {type(chunk)}")
            raise TypeError("Chunk must be a torch.Tensor")

        # Offload chunk if memory usage is high
        if self.chunk_offload_dir:
            chunk_offload_path = os.path.join(self.chunk_offload_dir, f"chunk_{chunk_index}.pt")
            if self._check_memory_usage():
                logger.info(f"High memory usage detected. Offloading chunk {chunk_index} to disk.")
                self._save_tensor(chunk, chunk_offload_path)
                loaded_chunk = self._load_tensor(chunk_offload_path, self.config.device)
                if loaded_chunk is not None:
                    chunk = loaded_chunk
                else:
                    logger.warning(f"Failed to load chunk from disk for chunk {chunk_index}, processing without offload.")
        
        # Ensure chunk is processed as a tensor
        if isinstance(chunk, list):
            chunk = torch.stack([torch.tensor(token) for token in chunk]).to(self.config.device)
        
        # Process each input in the chunk sequentially
        outputs = []
        for token in chunk:
            if isinstance(token, list):
                token = torch.tensor(token).to(self.config.device)
            outputs.append(self._predict_and_update(token))
        
        return torch.stack(outputs)

    def _check_memory_usage(self) -> bool:
        """
        Checks if current memory usage exceeds the configured threshold.

        Returns:
            True if memory usage is above threshold, else False.
        """
        mem_usage_gb = psutil.virtual_memory().used / (1024 ** 3)
        return mem_usage_gb > self.config.offload_threshold_gb

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """
        Processes an input sequence using chunk-wise processing and disk offloading.

        The input sequence is divided into chunks to manage memory usage. Each chunk
        is processed sequentially. If memory usage exceeds the defined threshold,
        the main memory tensors (memory_matrix and surprise_term) are offloaded to disk
        before processing the chunk and loaded back afterwards. Chunk sizes are
        dynamically adjusted based on memory pressure.

        Args:
            input_sequence: Tensor of shape (seq_len, input_dimension).

        Returns:
            Tensor of shape (seq_len, hidden_dimension) with outputs for each input token.
        """
        if not isinstance(input_sequence, torch.Tensor):
            logger.error(f"Expected torch.Tensor, but got {type(input_sequence)}")
            raise TypeError("Input sequence must be a torch.Tensor")
        if input_sequence.ndim != 2 or input_sequence.shape[1] != self.config.input_dimension:
            logger.error(f"Input sequence must have shape (seq_len, input_dimension), got {input_sequence.shape}")
            raise ValueError(f"Input sequence must have shape (seq_len, {self.config.input_dimension})")

        sequence_length = input_sequence.shape[0]
        outputs = []
        start_index = 0
        current_chunk_size = self.config.initial_chunk_size
        chunk_index = 0

        while start_index < sequence_length:
            end_index = min(start_index + current_chunk_size, sequence_length)
            chunk = input_sequence[start_index:end_index].to(self.config.device)

            # Check memory usage and offload main memory tensors if necessary
            if self._check_memory_usage():
                logger.info(f"High memory usage detected. Offloading memory for chunk starting at index {start_index}.")
                self._offload_to_disk()
                chunk_output = self._process_chunk(chunk, chunk_index)
                # Load back main memory tensors after processing the chunk
                self._load_from_disk()

                # Log a warning if loading failed, but proceed
                if self.memory_matrix is None or self.surprise_term is None:
                    logger.warning(f"Issue with loading memory/surprise term at chunk {chunk_index}.")
            else:
                chunk_output = self._process_chunk(chunk, chunk_index)

            outputs.append(chunk_output.cpu())
            start_index = end_index
            chunk_index += 1

            # Adjust chunk size based on current memory usage
            if self._check_memory_usage():
                # Reduce chunk size if memory is high
                current_chunk_size = max(self.config.min_chunk_size, current_chunk_size // 2)
                logger.debug(f"Adjusting chunk size due to high memory usage. New chunk size: {current_chunk_size}")
            else:
                # Increase chunk size if memory is low
                current_chunk_size = min(self.config.max_chunk_size, current_chunk_size * 2)
                logger.debug(f"Adjusting chunk size. New chunk size: {current_chunk_size}")

        return torch.cat(outputs, dim=0)

    def retrieve_memory(self, query: torch.Tensor) -> torch.Tensor:
        """
        Retrieves memory output for a given query without modifying the memory state.

        Args:
            query: Query tensor.

        Returns:
            Memory output tensor.
        """
        if not isinstance(query, torch.Tensor):
            logger.error(f"Expected torch.Tensor for query, but got {type(query)}")
            raise TypeError("Query must be a torch.Tensor")

        # Load offloaded tensors if necessary
        if self.memory_offloaded:
            self._load_from_disk()

        # Project the query
        projected_query = self._normalize_and_activate(self.query_projection(query.to(self.config.device)))

        # Retrieve memory output
        return self.memory_matrix @ projected_query

class TitanMemoryContext:
    """
    Context manager for TitanMemory to ensure proper cleanup.

    Usage:
        with TitanMemoryContext(config) as titan_memory:
            # Use titan_memory instance
            outputs = titan_memory(input_sequence)
    """
    def __init__(self, config: MemoryConfig):
        """Initializes the context manager with a MemoryConfig."""
        self.config = config
        self.titan_memory = None

    def __enter__(self):
        """Creates and returns a TitanMemory instance."""
        self.titan_memory = TitanMemory(self.config)
        return self.titan_memory

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensures cleanup is called when exiting the context."""
        try:
            if self.titan_memory:
                self.titan_memory.cleanup()
        except Exception as e:
            logger.error(f"Error during TitanMemory cleanup: {e}")

class TestTitanMemory(unittest.TestCase):
    def setUp(self):
        self.config = MemoryConfig(input_dimension=16, hidden_dimension=32, offload_threshold_gb=0.0001)
        self.memory = TitanMemory(self.config)
        self.deep_config = MemoryConfig(input_dimension=16, hidden_dimension=32, offload_threshold_gb=0.0001, use_deep_memory=True)
        self.deep_memory = TitanMemory(self.deep_config)
        self.adaptive_config = MemoryConfig(input_dimension=16, hidden_dimension=32, offload_threshold_gb=0.0001, adaptive_forgetting=True)
        self.adaptive_memory = TitanMemory(self.adaptive_config)

    def tearDown(self):
        self.memory.cleanup()
        self.deep_memory.cleanup()
        self.adaptive_memory.cleanup()

    def test_save_tensor(self):
        tensor = torch.randn(10, 10)
        test_path = os.path.join(tempfile.gettempdir(), "test_tensor.pt")
        self.memory._save_tensor(tensor, test_path)
        self.assertTrue(os.path.exists(test_path))
        try:
            os.remove(test_path)
        except OSError as e:
            logger.error(f"Error cleaning up test file {test_path}: {e}")

    def test_load_tensor(self):
        tensor = torch.randn(10, 10)
        test_path = os.path.join(tempfile.gettempdir(), "test_tensor.pt")
        torch.save(tensor, test_path)
        loaded_tensor = self.memory._load_tensor(test_path, torch.device('cpu'))
        self.assertIsNotNone(loaded_tensor)
        self.assertTrue(torch.equal(tensor, loaded_tensor))
        try:
            os.remove(test_path)
        except OSError as e:
            logger.error(f"Error cleaning up test file {test_path}: {e}")
        # Test loading non-existent file
        self.assertIsNone(self.memory._load_tensor("non_existent_path.pt", torch.device('cpu')))

    def test_offload_load_disk(self):
        initial_memory = self.memory.memory_matrix.clone()
        initial_surprise = self.memory.surprise_term.clone()
        self.memory._offload_to_disk()
        self.assertTrue(self.memory.memory_offloaded)
        self.assertTrue(self.memory.surprise_offloaded)
        self.assertIsNone(self.memory.memory_matrix)
        self.assertIsNone(self.memory.surprise_term)
        self.memory._load_from_disk()
        self.assertFalse(self.memory.memory_offloaded)
        self.assertFalse(self.memory.surprise_offloaded)
        self.assertIsNotNone(self.memory.memory_matrix)
        self.assertIsNotNone(self.memory.surprise_term)
        self.assertTrue(torch.equal(initial_memory, self.memory.memory_matrix))
        self.assertTrue(torch.equal(initial_surprise, self.memory.surprise_term))

    def test_normalize_and_activate(self):
        input_tensor = torch.randn(1, 10)
        output = self.memory._normalize_and_activate(input_tensor)
        self.assertEqual(output.shape, input_tensor.shape)
        self.assertTrue(torch.allclose(torch.linalg.norm(output, dim=-1), torch.ones(output.shape[:-1])))

    def test_compute_loss_and_gradient(self):
        prediction = torch.randn(self.config.hidden_dimension)
        value = torch.randn(self.config.hidden_dimension)
        key = torch.randn(self.config.hidden_dimension)
        loss, gradient = self.memory._compute_loss_and_gradient(prediction, value, key)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertIsInstance(gradient, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertEqual(gradient.shape, self.memory.memory_matrix.shape)

        # Test with incorrect shapes
        with self.assertRaises(ValueError):
            self.memory._compute_loss_and_gradient(prediction.unsqueeze(0), value, key)
        with self.assertRaises(ValueError):
            self.memory._compute_loss_and_gradient(prediction, value, key.unsqueeze(0))
        with self.assertRaises(TypeError):
            self.memory._compute_loss_and_gradient(prediction.tolist(), value, key)

    def test_update_memory(self):
        initial_memory = self.memory.memory_matrix.clone()
        gradient = torch.randn_like(self.memory.memory_matrix)
        self.memory._update_memory(gradient)
        self.assertFalse(torch.equal(initial_memory, self.memory.memory_matrix))

        # Test with incorrect gradient shape
        incorrect_gradient = torch.randn(self.config.hidden_dimension + 1, self.config.hidden_dimension + 1)
        with self.assertRaises(ValueError):
            self.memory._update_memory(incorrect_gradient)
        with self.assertRaises(TypeError):
            self.memory._update_memory(incorrect_gradient.tolist())

        # Test adaptive forgetting (basic check for no errors)
        self.memory._compute_adaptive_forgetting_rate(gradient)

    def test_deep_memory_update(self):
        initial_memory = self.deep_memory.memory_matrix.clone()
        gradient = torch.randn_like(self.deep_memory.memory_matrix)
        self.deep_memory._update_memory(gradient)
        self.assertFalse(torch.equal(initial_memory, self.deep_memory.memory_matrix))

    def test_adaptive_forgetting_update(self):
        initial_memory = self.adaptive_memory.memory_matrix.clone()
        gradient = torch.randn_like(self.adaptive_memory.memory_matrix)
        self.adaptive_memory._update_memory(gradient)
        self.assertFalse(torch.equal(initial_memory, self.adaptive_memory.memory_matrix))

    def test_retrieve_memory(self):
        query = torch.randn(self.config.input_dimension)
        output = self.memory.retrieve_memory(query)
        self.assertEqual(output.shape, (self.config.hidden_dimension,))

    def test_retrieve_memory_with_offloading(self):
        # Offload memory
        self.memory._offload_to_disk()
        self.assertTrue(self.memory.memory_offloaded)
        # Retrieve memory
        query = torch.randn(self.config.input_dimension)
        output = self.memory.retrieve_memory(query)
        self.assertEqual(output.shape, (self.config.hidden_dimension,))
        # Load back for cleanup
        self.memory._load_from_disk()

# Example usage:
if __name__ == "__main__":
    config = MemoryConfig(input_dimension=32, hidden_dimension=64, offload_threshold_gb=0.0001)
    with TitanMemoryContext(config) as model:
        sequence_length = 1024
        input_dimension = 32
        # Random input sequence
        input_sequence = torch.randn(sequence_length, input_dimension)
        outputs = model(input_sequence)
        print("Output shape:", outputs.shape)  # Expected: (1024, 64)

    deep_config = MemoryConfig(input_dimension=32, hidden_dimension=64, offload_threshold_gb=0.0001, use_deep_memory=True)
    with TitanMemoryContext(deep_config) as deep_model:
        outputs = deep_model(input_sequence)
        print("Deep Memory Output shape:", outputs.shape)

    adaptive_config = MemoryConfig(input_dimension=32, hidden_dimension=64, offload_threshold_gb=0.0001, adaptive_forgetting=True)
    with TitanMemoryContext(adaptive_config) as adaptive_model:
        outputs = adaptive_model(input_sequence)
        print("Adaptive Memory Output shape:", outputs.shape)

    # Example of using retrieve_memory
    query_vector = torch.randn(input_dimension)
    memory_output = model.retrieve_memory(query_vector)
    print("Retrieved memory output shape:", memory_output.shape)

    # Run unit tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
