import abc
import time
from typing import (
import numpy as np
from ray.data._internal.block_batching.iter_batches import iter_batches
from ray.data._internal.stats import DatasetStats, StatsManager
from ray.data.block import (
from ray.types import ObjectRef
from ray.util.annotations import PublicAPI
def validate_columns(columns: Union[str, List]) -> None:
    if isinstance(columns, list):
        for column in columns:
            validate_column(column)
    else:
        validate_column(columns)