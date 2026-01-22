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
def _get_num_runs_launched(self, runs: List[Dict[str, Any]]) -> int:
    """Returns the number of valid runs in the sweep."""
    count = 0
    for run in runs:
        if run.get('state', '') in ['killed', 'crashed'] and (not run.get('summaryMetrics')):
            _logger.debug(f'excluding run: {run['name']} with state: {run['state']} from run cap \n{run}')
            continue
        count += 1
    return count