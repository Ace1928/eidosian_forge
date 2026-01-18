import warnings
from typing import List
from unittest.mock import Mock
import torch
from torch.utils.data import DataLoader, IterableDataset, TensorDataset
from accelerate.accelerator import Accelerator, DataLoaderConfiguration
from accelerate.utils.dataclasses import DistributedType
def verify_dataloader_batch_sizes(accelerator: Accelerator, dataset_size: int, batch_size: int, process_0_expected_batch_sizes: List[int], process_1_expected_batch_sizes: List[int]):
    """
    A helper function for verifying the batch sizes coming from a prepared dataloader in each process
    """
    dl = create_dataloader(accelerator=accelerator, dataset_size=dataset_size, batch_size=batch_size)
    batch_sizes = [len(batch[0]) for batch in dl]
    if accelerator.process_index == 0:
        assert batch_sizes == process_0_expected_batch_sizes
    elif accelerator.process_index == 1:
        assert batch_sizes == process_1_expected_batch_sizes