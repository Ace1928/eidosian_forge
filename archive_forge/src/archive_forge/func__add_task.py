import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Generator, Optional, Union, cast
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.progress_bar import ProgressBar
from pytorch_lightning.utilities.types import STEP_OUTPUT
def _add_task(self, total_batches: Union[int, float], description: str, visible: bool=True) -> 'TaskID':
    assert self.progress is not None
    return self.progress.add_task(f'[{self.theme.description}]{description}', total=total_batches, visible=visible)