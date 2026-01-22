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
def _check_run_exists_and_inited(self, entity: str, project: str, run_id: str, rqi_id: str) -> bool:
    """Checks the stateof the run to ensure it has been inited. Note this will not behave well with resuming."""
    try:
        run_state = self._api.get_run_state(entity, project, run_id)
        if run_state.lower() != 'pending':
            return True
    except CommError:
        _logger.info(f'Run {entity}/{project}/{run_id} with rqi id: {rqi_id} did not have associated run')
    return False