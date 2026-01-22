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
def _configure_model_summary_callback(self, enable_model_summary: bool) -> None:
    if not enable_model_summary:
        return
    model_summary_cbs = [type(cb) for cb in self.trainer.callbacks if isinstance(cb, ModelSummary)]
    if model_summary_cbs:
        rank_zero_info(f'Trainer already configured with model summary callbacks: {model_summary_cbs}. Skipping setting a default `ModelSummary` callback.')
        return
    progress_bar_callback = self.trainer.progress_bar_callback
    is_progress_bar_rich = isinstance(progress_bar_callback, RichProgressBar)
    model_summary: ModelSummary
    if progress_bar_callback is not None and is_progress_bar_rich:
        model_summary = RichModelSummary()
    else:
        model_summary = ModelSummary()
    self.trainer.callbacks.append(model_summary)