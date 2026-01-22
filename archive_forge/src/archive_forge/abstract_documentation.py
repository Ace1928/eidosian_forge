import logging
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dockerpycreds.utils import find_executable  # type: ignore
import wandb
from wandb.apis.internal import Api
from wandb.sdk.lib import runid
from .._project_spec import LaunchProject
Submit an LaunchProject to be run.

        Returns a SubmittedRun object to track the execution
        Arguments:
        launch_project: Object of _project_spec.LaunchProject class representing a wandb launch project

        Returns:
            A :py:class:`wandb.sdk.launch.runners.SubmittedRun`. This function is expected to run
            the project asynchronously, i.e. it should trigger project execution and then
            immediately return a `SubmittedRun` to track execution status.
        