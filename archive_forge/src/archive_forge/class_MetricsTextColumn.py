import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Generator, Optional, Union, cast
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.progress_bar import ProgressBar
from pytorch_lightning.utilities.types import STEP_OUTPUT
class MetricsTextColumn(ProgressColumn):
    """A column containing text."""

    def __init__(self, trainer: 'pl.Trainer', style: Union[str, 'Style'], text_delimiter: str, metrics_format: str):
        self._trainer = trainer
        self._tasks: Dict[Union[int, TaskID], Any] = {}
        self._current_task_id = 0
        self._metrics: Dict[Union[str, 'Style'], Any] = {}
        self._style = style
        self._text_delimiter = text_delimiter
        self._metrics_format = metrics_format
        super().__init__()

    def update(self, metrics: Dict[Any, Any]) -> None:
        self._metrics = metrics

    def render(self, task: 'Task') -> Text:
        assert isinstance(self._trainer.progress_bar_callback, RichProgressBar)
        if self._trainer.state.fn != 'fit' or self._trainer.sanity_checking or self._trainer.progress_bar_callback.train_progress_bar_id != task.id:
            return Text()
        if self._trainer.training and task.id not in self._tasks:
            self._tasks[task.id] = 'None'
            if self._renderable_cache:
                self._current_task_id = cast(TaskID, self._current_task_id)
                self._tasks[self._current_task_id] = self._renderable_cache[self._current_task_id][1]
            self._current_task_id = task.id
        if self._trainer.training and task.id != self._current_task_id:
            return self._tasks[task.id]
        metrics_texts = self._generate_metrics_texts()
        text = self._text_delimiter.join(metrics_texts)
        return Text(text, justify='left', style=self._style)

    def _generate_metrics_texts(self) -> Generator[str, None, None]:
        for name, value in self._metrics.items():
            if not isinstance(value, str):
                value = f'{value:{self._metrics_format}}'
            yield f'{name}: {value}'