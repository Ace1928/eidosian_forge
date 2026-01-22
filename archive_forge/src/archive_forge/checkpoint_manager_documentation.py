import logging
import numbers
from typing import Any, Callable, List, Optional, Tuple
from ray._private.dict import flatten_dict
from ray.air._internal.util import is_nan
from ray.air.config import MAX
from ray.train import CheckpointConfig
from ray.train._internal.session import _TrainingResult
from ray.train._internal.storage import _delete_fs_path
Get the score for a checkpoint, according to checkpoint config.

        If `mode="min"`, the metric is negated so that the lowest score is
        treated as the best.

        Returns:
            Tuple: A tuple of (not_is_nan: bool, score: numbers.Number).
                This score orders: nan values < float("-inf") < valid numeric metrics
        