from enum import Enum
from typing import List, Optional
from pydantic import (  # type: ignore
import wandb
from wandb.sdk.launch.utils import (
@root_validator(pre=True)
@classmethod
def validate_docker(cls, values: dict) -> dict:
    """Right now there are no required fields for docker builds."""
    return values