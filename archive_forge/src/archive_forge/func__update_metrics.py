import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Generator, Optional, Union, cast
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.progress_bar import ProgressBar
from pytorch_lightning.utilities.types import STEP_OUTPUT
def _update_metrics(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
    metrics = self.get_metrics(trainer, pl_module)
    if self._metric_component:
        self._metric_component.update(metrics)