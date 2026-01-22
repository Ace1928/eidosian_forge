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
def _run_init(self, init: AutologInitArgs=None) -> None:
    """Handle wandb run initialization."""
    if init:
        _wandb_run = wandb.run
        self._run = wandb.init(**init)
        if _wandb_run != self._run:
            self.__run_created_by_autolog = True
    elif wandb.run is None:
        self._run = wandb.init()
        self.__run_created_by_autolog = True
    else:
        self._run = wandb.run