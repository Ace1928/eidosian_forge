import inspect
import functools
from enum import Enum
import torch.autograd
class _SnapshotState(Enum):
    """
    These are the snapshotting-related states that IterDataPipes can be in.

    `NotStarted` - allows you to restore a snapshot and create an iterator with reset
    `Restored` - cannot restore again, allows you to create an iterator without resetting the DataPipe
    `Iterating` - can restore, will reset if you create a new iterator
    """
    NotStarted = 0
    Restored = 1
    Iterating = 2