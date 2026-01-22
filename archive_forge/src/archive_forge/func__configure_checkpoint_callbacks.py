import logging
import os
from datetime import timedelta
from typing import Dict, List, Optional, Sequence, Union
import pytorch_lightning as pl
from lightning_fabric.utilities.registry import _load_external_callbacks
from pytorch_lightning.callbacks import (
from pytorch_lightning.callbacks.batch_size_finder import BatchSizeFinder
from pytorch_lightning.callbacks.lr_finder import LearningRateFinder
from pytorch_lightning.callbacks.rich_model_summary import RichModelSummary
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.trainer import call
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_info
def _configure_checkpoint_callbacks(self, enable_checkpointing: bool) -> None:
    if self.trainer.checkpoint_callbacks:
        if not enable_checkpointing:
            raise MisconfigurationException('Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list.')
    elif enable_checkpointing:
        self.trainer.callbacks.append(ModelCheckpoint())