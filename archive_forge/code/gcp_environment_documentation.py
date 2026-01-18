import logging
import os
import subprocess
from typing import Optional
from wandb.sdk.launch.errors import LaunchError
from wandb.util import get_module
from ..utils import GCS_URI_RE, event_loop_thread_exec
from .abstract import AbstractEnvironment
Upload a directory to GCS.

        Arguments:
            source: The path to the local directory.
            destination: The path to the GCS directory.

        Raises:
            LaunchError: If the directory cannot be uploaded.
        