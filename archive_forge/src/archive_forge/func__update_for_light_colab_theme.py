import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Generator, Optional, Union, cast
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.progress_bar import ProgressBar
from pytorch_lightning.utilities.types import STEP_OUTPUT
def _update_for_light_colab_theme(self) -> None:
    if _detect_light_colab_theme():
        attributes = ['description', 'batch_progress', 'metrics']
        for attr in attributes:
            if getattr(self.theme, attr) == 'white':
                setattr(self.theme, attr, 'black')