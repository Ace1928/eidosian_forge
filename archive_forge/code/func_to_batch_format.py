import collections
import os
import time
from dataclasses import dataclass
from typing import (
import numpy as np
import ray
from ray import DynamicObjectRefGenerator
from ray.data._internal.util import _check_pyarrow_version, _truncated_repr
from ray.types import ObjectRef
from ray.util.annotations import DeveloperAPI
import psutil
def to_batch_format(self, batch_format: Optional[str]) -> DataBatch:
    """Convert this block into the provided batch format.

        Args:
            batch_format: The batch format to convert this block to.

        Returns:
            This block formatted as the provided batch format.
        """
    if batch_format is None:
        return self.to_block()
    elif batch_format == 'default' or batch_format == 'native':
        return self.to_default()
    elif batch_format == 'pandas':
        return self.to_pandas()
    elif batch_format == 'pyarrow':
        return self.to_arrow()
    elif batch_format == 'numpy':
        return self.to_numpy()
    else:
        raise ValueError(f'The batch format must be one of {VALID_BATCH_FORMATS}, got: {batch_format}')