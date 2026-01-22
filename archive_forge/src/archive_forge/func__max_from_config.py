import asyncio
import logging
import os
import pprint
import threading
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Event
from typing import Any, Dict, List, Optional, Union
import wandb
from wandb.apis.internal import Api
from wandb.errors import CommError
from wandb.sdk.launch._launch_add import launch_add
from wandb.sdk.launch.runner.local_container import LocalSubmittedRun
from wandb.sdk.launch.runner.local_process import LocalProcessRunner
from wandb.sdk.launch.sweeps.scheduler import Scheduler
from wandb.sdk.lib import runid
from .. import loader
from .._project_spec import LaunchProject
from ..builder.build import construct_agent_configs
from ..errors import LaunchDockerError, LaunchError
from ..utils import (
from .job_status_tracker import JobAndRunStatusTracker
from .run_queue_item_file_saver import RunQueueItemFileSaver
def _max_from_config(config: Dict[str, Any], key: str, default: int=1) -> Union[int, float]:
    """Get an integer from the config, or float.inf if -1.

    Utility for parsing integers from the agent config with a default, infinity
    handling, and integer parsing. Raises more informative error if parse error.
    """
    try:
        val = config.get(key)
        if val is None:
            val = default
        max_from_config = int(val)
    except ValueError as e:
        raise LaunchError(f"Error when parsing LaunchAgent config key: ['{key}': {config.get(key)}]. Error: {str(e)}")
    if max_from_config == -1:
        return float('inf')
    if max_from_config < 0:
        raise LaunchError(f"Error when parsing LaunchAgent config key: ['{key}': {config.get(key)}]. Error: negative value.")
    return max_from_config