# core.py
import torch
from torch.utils.data import DataLoader, Dataset
import psutil
import logging
import threading
from typing import Callable, Optional, Dict, Any, Tuple
import os
import tempfile

from eidos_config import DEFAULT_ADAPTIVE_CHUNKING, 
from eidos_logging import configure_logging
from eidos_files import 

configure_logging()
# Initialize logging
logger = logging.getLogger(__name__)


class CPUDataset(Dataset):
    """
    A dataset designed for CPU-based training, allowing for data to be loaded
    in chunks and potentially offloaded to disk.
    """

    def __init__(
        self,
        data_source: Any,
        config: LLMConfig,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            data_source (Any): The original data source (e.g., a list, a file path).
            config (LLMConfig): The Eidos configuration object.
            transform (Optional[Callable]): An optional transform function to apply to the data.
        """
        self.data_source = data_source
        self.transform = transform
        self.chunk_size = config.initial_chunk_size
        self.offload_to_disk = True  # Always offload to disk for large datasets
        self.temp_dir = create_temporary_directory("cpu_dataset_chunks")
        self._create_chunks()
        self._lock = threading.Lock()  # Lock for thread-safe access to chunks
        logger.info(
            f"CPUDataset initialized with chunk size: {self.chunk_size}, offloading to: {self.temp_dir}"
        )

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        with self._lock:
            chunk_index = idx // self.chunk_size
            index_within_chunk = idx % self.chunk_size
            chunk = self._load_chunk(chunk_index)
            sample = chunk[index_within_chunk]
            if self.transform:
                sample = self.transform(sample)
            return sample


class CPUTrainer:
    """
    A trainer designed for CPU-only training with dynamic chunking and resource monitoring.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: CPUDataset,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        config: LLMConfig,
    ):
        """
        Args:
            model (torch.nn.Module): The neural network model to train.
            dataset (CPUDataset): The dataset to use for training.
            optimizer (torch.optim.Optimizer): The optimization algorithm.
            loss_fn (Callable): The loss function.
            config (LLMConfig): The Eidos configuration object.
        """
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.batch_size = config.training_batch_size
        self.num_epochs = config.training_num_epochs
        self.learning_rate = config.training_learning_rate
        self.resource_check_interval = config.resource_check_interval
        self.high_memory_threshold = config.high_memory_threshold
        self.low_memory_threshold = config.low_memory_threshold
        self.chunk_size_reduction_factor = config.chunk_size_reduction_factor
        self.chunk_size_increase_factor = config.chunk_size_increase_factor
        self.min_chunk_size = config.min_chunk_size
        self.max_chunk_size = config.max_chunk_size
        self.use_threading = config.use_data_loader_threading
        self.log_interval = config.training_log_interval
        self.adaptive_chunking = config.adaptive_chunking
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.running = False
        self.resource_monitor_thread = None
        logger.info(
            f"CPUTrainer initialized with batch size: {self.batch_size}, epochs: {self.num_epochs}, adaptive chunking: {self.adaptive_chunking}"
        )

    def train_epoch(self, epoch):
        """
        Trains the model for one epoch.
        """
        num_workers = os.cpu_count() if self.use_threading else 0
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,  # Helps in faster data transfer to CPU
        )

        self.model.train()
        for batch_idx, batch in enumerate(dataloader):
            self.optimizer.zero_grad()
            try:
                inputs, targets = (
                    batch[:-1],
                    batch[-1],
                )  # Assuming last element is the target
                inputs = tuple(inp.to(self.device) for inp in inputs)
                targets = targets.to(self.device)
                outputs = self.model(*inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

                if batch_idx % self.log_interval == 0:
                    logger.info(
                        f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}"
                    )
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")

    def monitor_resources(self):
        """
        Monitors CPU and memory usage and adjusts chunk size dynamically.
        """
        while self.running and self.adaptive_chunking:
            try:
                memory_percent = psutil.virtual_memory().percent
                logger.debug(f"Memory usage: {memory_percent}%")

                if (
                    memory_percent > self.high_memory_threshold * 100
                    and self.dataset.chunk_size > self.min_chunk_size
                ):
                    new_chunk_size = max(
                        self.min_chunk_size,
                        self.dataset.chunk_size // self.chunk_size_reduction_factor,
                    )
                    self.dataset.set_chunk_size(new_chunk_size)
                elif (
                    memory_percent < self.low_memory_threshold * 100
                    and self.dataset.chunk_size < self.max_chunk_size
                ):
                    new_chunk_size = min(
                        self.max_chunk_size,
                        self.dataset.chunk_size * self.chunk_size_increase_factor,
                    )
                    self.dataset.set_chunk_size(new_chunk_size)

                torch.cuda.empty_cache()
                torch.cpu.empty_cache()
            except Exception as e:
                logger.error(f"Error monitoring resources: {e}")
            finally:
                import time

                time.sleep(self.resource_check_interval)

    def train(self):
        """
        Starts the training process with resource monitoring.
        """
        self.running = True
        if self.adaptive_chunking:
            self.resource_monitor_thread = threading.Thread(
                target=self.monitor_resources
            )
            self.resource_monitor_thread.start()
            logger.info("Resource monitoring started.")

        logger.info("Starting training...")
        for epoch in range(self.num_epochs):
            self.train_epoch(epoch)
        logger.info("Training finished.")
        self.stop_monitoring()
        self.dataset.cleanup()

    def stop_monitoring(self):
        """
        Stops the resource monitoring thread.
        """
        self.running = False
        if self.resource_monitor_thread and self.resource_monitor_thread.is_alive():
            self.resource_monitor_thread.join()
            logger.info("Resource monitoring stopped.")
