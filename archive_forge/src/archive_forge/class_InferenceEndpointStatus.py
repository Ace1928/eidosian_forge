import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional
from .inference._client import InferenceClient
from .inference._generated._async_client import AsyncInferenceClient
from .utils import logging, parse_datetime
class InferenceEndpointStatus(str, Enum):
    PENDING = 'pending'
    INITIALIZING = 'initializing'
    UPDATING = 'updating'
    UPDATE_FAILED = 'updateFailed'
    RUNNING = 'running'
    PAUSED = 'paused'
    FAILED = 'failed'
    SCALED_TO_ZERO = 'scaledToZero'