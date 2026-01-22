import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Generator, Optional, Union, cast
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.progress_bar import ProgressBar
from pytorch_lightning.utilities.types import STEP_OUTPUT
def _get_train_description(self, current_epoch: int) -> str:
    train_description = f'Epoch {current_epoch}'
    if self.trainer.max_epochs is not None:
        train_description += f'/{self.trainer.max_epochs - 1}'
    if len(self.validation_description) > len(train_description):
        train_description = f'{train_description:{len(self.validation_description)}}'
    return train_description