import asyncio
import base64
import copy
import logging
import os
import socket
import threading
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple, Union
import click
import yaml
import wandb
from wandb.errors import CommError
from wandb.sdk.launch._launch_add import launch_add
from wandb.sdk.launch.errors import LaunchError
from wandb.sdk.launch.sweeps import SchedulerError
from wandb.sdk.launch.sweeps.utils import (
from wandb.sdk.launch.utils import (
from wandb.sdk.lib.runid import generate_id
def _stop_run(self, run_id: str) -> bool:
    """Stops a run and removes it from the scheduler."""
    if run_id not in self._runs:
        _logger.debug(f'run: {run_id} not in _runs: {self._runs}')
        return False
    run = self._runs[run_id]
    del self._runs[run_id]
    if not run.queued_run:
        _logger.debug(f'tried to _stop_run but run not queued yet (run_id:{run.id})')
        return False
    if not run.state.is_alive:
        return True
    encoded_run_id = base64.standard_b64encode(f'Run:v1:{run_id}:{self._project}:{self._entity}'.encode()).decode('utf-8')
    try:
        success: bool = self._api.stop_run(run_id=encoded_run_id)
        if success:
            wandb.termlog(f'{LOG_PREFIX}Stopped run {run_id}.')
            return True
    except Exception as e:
        _logger.debug(f'error stopping run ({run_id}): {e}')
    return False