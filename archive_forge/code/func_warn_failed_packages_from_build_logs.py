import asyncio
import json
import logging
import os
import platform
import re
import subprocess
import sys
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast
import click
import wandb
import wandb.docker as docker
from wandb import util
from wandb.apis.internal import Api
from wandb.errors import CommError
from wandb.sdk.launch.errors import LaunchError
from wandb.sdk.launch.git_reference import GitReference
from wandb.sdk.launch.wandb_reference import WandbReference
from wandb.sdk.wandb_config import Config
from .builder.templates._wandb_bootstrap import (
def warn_failed_packages_from_build_logs(log: str, image_uri: str, api: Api, job_tracker: Optional['JobAndRunStatusTracker']) -> None:
    match = FAILED_PACKAGES_REGEX.search(log)
    if match:
        _msg = f'Failed to install the following packages: {match.group(1)} for image: {image_uri}. Will attempt to launch image without them.'
        wandb.termwarn(_msg)
        if job_tracker is not None:
            res = job_tracker.saver.save_contents(_msg, 'failed-packages.log', 'warning')
            api.update_run_queue_item_warning(job_tracker.run_queue_item_id, 'Some packages were not successfully installed during the build', 'build', res)