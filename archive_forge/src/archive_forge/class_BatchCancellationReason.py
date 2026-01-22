from __future__ import absolute_import
import abc
import enum
import typing
from typing import Optional, Sequence
class BatchCancellationReason(str, enum.Enum):
    """An enum-like class representing reasons why a batch was cancelled."""
    PRIOR_ORDERED_MESSAGE_FAILED = 'Batch cancelled because prior ordered message for the same key has failed. This batch has been cancelled to avoid out-of-order publish.'
    CLIENT_STOPPED = 'Batch cancelled because the publisher client has been stopped.'