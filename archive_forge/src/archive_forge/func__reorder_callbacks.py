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
@staticmethod
def _reorder_callbacks(callbacks: List[Callback]) -> List[Callback]:
    """Moves all the tuner specific callbacks at the beginning of the list and all the `ModelCheckpoint` callbacks
        to the end of the list. The sequential order within the group of checkpoint callbacks is preserved, as well as
        the order of all other callbacks.

        Args:
            callbacks: A list of callbacks.

        Return:
            A new list in which the first elements are tuner specific callbacks and last elements are ModelCheckpoints
            if there were any present in the input.

        """
    tuner_callbacks: List[Callback] = []
    other_callbacks: List[Callback] = []
    checkpoint_callbacks: List[Callback] = []
    for cb in callbacks:
        if isinstance(cb, (BatchSizeFinder, LearningRateFinder)):
            tuner_callbacks.append(cb)
        elif isinstance(cb, Checkpoint):
            checkpoint_callbacks.append(cb)
        else:
            other_callbacks.append(cb)
    return tuner_callbacks + other_callbacks + checkpoint_callbacks