from enum import Enum
from typing import List, Optional
from pydantic import (  # type: ignore
import wandb
from wandb.sdk.launch.utils import (
class BuilderType(str, Enum):
    """Enum of valid builder types."""
    docker = 'docker'
    kaniko = 'kaniko'
    noop = 'noop'