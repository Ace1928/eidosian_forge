import asyncio
import json
import time
from dataclasses import dataclass, replace, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from ray._private import ray_constants
from ray._private.gcs_utils import GcsAioClient
from ray._private.runtime_env.packaging import parse_uri
from ray.experimental.internal_kv import (
from ray.util.annotations import PublicAPI
@PublicAPI(stability='stable')
class JobStatus(str, Enum):
    """An enumeration for describing the status of a job."""
    PENDING = 'PENDING'
    RUNNING = 'RUNNING'
    STOPPED = 'STOPPED'
    SUCCEEDED = 'SUCCEEDED'
    FAILED = 'FAILED'

    def __str__(self) -> str:
        return f'{self.value}'

    def is_terminal(self) -> bool:
        """Return whether or not this status is terminal.

        A terminal status is one that cannot transition to any other status.
        The terminal statuses are "STOPPED", "SUCCEEDED", and "FAILED".

        Returns:
            True if this status is terminal, otherwise False.
        """
        return self.value in {'STOPPED', 'SUCCEEDED', 'FAILED'}