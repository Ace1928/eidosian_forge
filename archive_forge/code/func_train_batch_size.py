import contextlib
import io
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from huggingface_hub import get_full_repo_name
from packaging import version
from .debug_utils import DebugOption
from .trainer_utils import (
from .utils import (
from .utils.generic import strtobool
from .utils.import_utils import is_optimum_neuron_available
@property
def train_batch_size(self) -> int:
    """
        The actual batch size for training (may differ from `per_gpu_train_batch_size` in distributed training).
        """
    if self.per_gpu_train_batch_size:
        logger.warning('Using deprecated `--per_gpu_train_batch_size` argument which will be removed in a future version. Using `--per_device_train_batch_size` is preferred.')
    per_device_batch_size = self.per_gpu_train_batch_size or self.per_device_train_batch_size
    train_batch_size = per_device_batch_size * max(1, self.n_gpu)
    return train_batch_size