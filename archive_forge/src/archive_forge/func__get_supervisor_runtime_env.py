import asyncio
import copy
import json
import logging
import os
import psutil
import random
import signal
import string
import subprocess
import sys
import time
import traceback
from asyncio.tasks import FIRST_COMPLETED
from collections import deque
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from ray.util.scheduling_strategies import (
import ray
from ray._private.gcs_utils import GcsAioClient
from ray._private.utils import run_background_task
import ray._private.ray_constants as ray_constants
from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR
from ray.actor import ActorHandle
from ray.dashboard.consts import (
from ray.dashboard.modules.job.common import (
from ray.dashboard.modules.job.utils import file_tail_iterator
from ray.exceptions import ActorUnschedulableError, RuntimeEnvSetupError
from ray.job_submission import JobStatus
from ray._private.event.event_logger import get_event_logger
from ray.core.generated.event_pb2 import Event
def _get_supervisor_runtime_env(self, user_runtime_env: Dict[str, Any], resources_specified: bool=False) -> Dict[str, Any]:
    """Configure and return the runtime_env for the supervisor actor.

        Args:
            user_runtime_env: The runtime_env specified by the user.
            resources_specified: Whether the user specified resources in the
                submit_job() call. If so, we will skip the workaround introduced
                in #24546 for GPU detection and just use the user's resource
                requests, so that the behavior matches that of the user specifying
                resources for any other actor.

        Returns:
            The runtime_env for the supervisor actor.
        """
    runtime_env = copy.deepcopy(user_runtime_env) if user_runtime_env is not None else {}
    env_vars = runtime_env.get('env_vars')
    if env_vars is None:
        env_vars = {}
    env_vars[ray_constants.RAY_WORKER_NICENESS] = '0'
    if not resources_specified:
        env_vars[ray_constants.NOSET_CUDA_VISIBLE_DEVICES_ENV_VAR] = '1'
    runtime_env['env_vars'] = env_vars
    return runtime_env