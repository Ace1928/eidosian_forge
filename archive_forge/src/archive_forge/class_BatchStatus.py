from __future__ import absolute_import
import abc
import enum
import typing
from typing import Optional, Sequence
class BatchStatus(str, enum.Enum):
    """An enum-like class representing valid statuses for a batch."""
    ACCEPTING_MESSAGES = 'accepting messages'
    STARTING = 'starting'
    IN_PROGRESS = 'in progress'
    ERROR = 'error'
    SUCCESS = 'success'