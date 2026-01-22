import abc
from dataclasses import dataclass
from typing import Any, List
from ray.data.block import Block, DataBatch
from ray.types import ObjectRef
class CollatedBatch(Batch):
    """A batch of collated data with a corresponding index.

    Attributes:
        batch_idx: The global index of this batch so that downstream operations can
            maintain ordering.
        data: The batch of data which is the output of a user provided collate_fn
            Therefore, the type of this data can be Any.
    """
    batch_idx: int
    data: Any