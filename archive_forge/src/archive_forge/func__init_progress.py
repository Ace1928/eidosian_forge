import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Generator, Optional, Union, cast
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.progress_bar import ProgressBar
from pytorch_lightning.utilities.types import STEP_OUTPUT
def _init_progress(self, trainer: 'pl.Trainer') -> None:
    if self.is_enabled and (self.progress is None or self._progress_stopped):
        self._reset_progress_bar_ids()
        reconfigure(**self._console_kwargs)
        self._console = get_console()
        self._console.clear_live()
        self._metric_component = MetricsTextColumn(trainer, self.theme.metrics, self.theme.metrics_text_delimiter, self.theme.metrics_format)
        self.progress = CustomProgress(*self.configure_columns(trainer), self._metric_component, auto_refresh=False, disable=self.is_disabled, console=self._console)
        self.progress.start()
        self._progress_stopped = False