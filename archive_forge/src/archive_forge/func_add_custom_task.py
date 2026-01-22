import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Generator, Optional, Union, cast
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.progress_bar import ProgressBar
from pytorch_lightning.utilities.types import STEP_OUTPUT
def add_custom_task(self, task: CustomInfiniteTask, start: bool=True) -> TaskID:
    with self._lock:
        self._tasks[self._task_index] = task
        if start:
            self.start_task(self._task_index)
        new_task_index = self._task_index
        self._task_index = TaskID(int(self._task_index) + 1)
    self.refresh()
    return new_task_index