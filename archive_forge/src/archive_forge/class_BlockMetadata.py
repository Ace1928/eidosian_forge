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
@DeveloperAPI
@dataclass
class BlockMetadata:
    """Metadata about the block."""
    num_rows: Optional[int]
    size_bytes: Optional[int]
    schema: Optional[Union[type, 'pyarrow.lib.Schema']]
    input_files: Optional[List[str]]
    exec_stats: Optional[BlockExecStats]

    def __post_init__(self):
        if self.input_files is None:
            self.input_files = []
        if self.size_bytes is not None:
            assert isinstance(self.size_bytes, int)