import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Generator, Optional, Union, cast
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.progress_bar import ProgressBar
from pytorch_lightning.utilities.types import STEP_OUTPUT
def _generate_metrics_texts(self) -> Generator[str, None, None]:
    for name, value in self._metrics.items():
        if not isinstance(value, str):
            value = f'{value:{self._metrics_format}}'
        yield f'{name}: {value}'