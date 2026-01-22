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
def _validate_callbacks_list(callbacks: List[Callback]) -> None:
    stateful_callbacks = [cb for cb in callbacks if is_overridden('state_dict', instance=cb)]
    seen_callbacks = set()
    for callback in stateful_callbacks:
        if callback.state_key in seen_callbacks:
            raise RuntimeError(f'Found more than one stateful callback of type `{type(callback).__name__}`. In the current configuration, this callback does not support being saved alongside other instances of the same type. Please consult the documentation of `{type(callback).__name__}` regarding valid settings for the callback state to be checkpointable. HINT: The `callback.state_key` must be unique among all callbacks in the Trainer.')
        seen_callbacks.add(callback.state_key)