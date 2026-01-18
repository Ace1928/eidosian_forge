import asyncio
import logging
from typing import Any, Dict, List, Optional, cast
import wandb
from wandb.apis.internal import Api
from wandb.sdk.launch.environment.aws_environment import AwsEnvironment
from wandb.sdk.launch.errors import LaunchError
from .._project_spec import EntryPoint, LaunchProject, get_entry_point_command
from ..builder.build import get_env_vars_dict
from ..registry.abstract import AbstractRegistry
from ..utils import (
from .abstract import AbstractRun, AbstractRunner, Status
Run a project on Amazon Sagemaker.

        Arguments:
            launch_project (LaunchProject): The project to run.

        Returns:
            Optional[AbstractRun]: The run instance.

        Raises:
            LaunchError: If the launch is unsuccessful.
        