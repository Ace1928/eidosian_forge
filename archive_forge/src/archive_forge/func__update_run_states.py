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
def _update_run_states(self) -> None:
    """Iterate through runs.

        Get state from backend and deletes runs if not in running state. Threadsafe.
        """
    runs_to_remove: List[str] = []
    for run_id, run in self._yield_runs():
        run.state = self._get_run_state(run_id, run.state)
        try:
            rqi_state = run.queued_run.state if run.queued_run else None
        except (CommError, LaunchError) as e:
            _logger.debug(f'Failed to get queued_run.state: {e}')
            rqi_state = None
        if not run.state.is_alive or rqi_state == 'failed':
            _logger.debug(f'({run_id}) states: ({run.state}, {rqi_state})')
            runs_to_remove.append(run_id)
    self._cleanup_runs(runs_to_remove)