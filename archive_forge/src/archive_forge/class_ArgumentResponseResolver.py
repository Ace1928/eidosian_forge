import asyncio
import functools
import inspect
import logging
import sys
from typing import Any, Dict, Optional, Sequence, TypeVar
import wandb.sdk
import wandb.util
from wandb.sdk.lib import telemetry as wb_telemetry
from wandb.sdk.lib.timer import Timer
class ArgumentResponseResolver(Protocol):

    def __call__(self, args: Sequence[Any], kwargs: Dict[str, Any], response: Response, start_time: float, time_elapsed: float) -> Optional[Dict[str, Any]]:
        ...