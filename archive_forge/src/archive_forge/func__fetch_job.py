import enum
import logging
import os
import tempfile
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast
import wandb
import wandb.docker as docker
from wandb.apis.internal import Api
from wandb.errors import CommError
from wandb.sdk.launch import utils
from wandb.sdk.lib.runid import generate_id
from .errors import LaunchError
from .utils import LOG_PREFIX, recursive_macro_sub
def _fetch_job(self) -> None:
    """Fetches the job details from the public API and configures the launch project.

        Raises:
            LaunchError: If there is an error accessing the job.
        """
    public_api = wandb.apis.public.Api()
    job_dir = tempfile.mkdtemp()
    try:
        job = public_api.job(self.job, path=job_dir)
    except CommError as e:
        msg = e.message
        raise LaunchError(f'Error accessing job {self.job}: {msg} on {public_api.settings.get('base_url')}')
    job.configure_launch_project(self)
    self._job_artifact = job._job_artifact