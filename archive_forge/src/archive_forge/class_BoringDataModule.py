from typing import Any, Dict, Iterator, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset
from lightning_fabric.utilities.types import _TORCH_LRSCHEDULER
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.utilities.types import STEP_OUTPUT
class BoringDataModule(LightningDataModule):
    """
    .. warning::  This is meant for testing/debugging and is experimental.
    """

    def __init__(self) -> None:
        super().__init__()
        self.random_full = RandomDataset(32, 64 * 4)

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.random_train = Subset(self.random_full, indices=range(64))
        if stage in ('fit', 'validate'):
            self.random_val = Subset(self.random_full, indices=range(64, 64 * 2))
        if stage == 'test':
            self.random_test = Subset(self.random_full, indices=range(64 * 2, 64 * 3))
        if stage == 'predict':
            self.random_predict = Subset(self.random_full, indices=range(64 * 3, 64 * 4))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.random_train)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.random_val)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.random_test)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.random_predict)