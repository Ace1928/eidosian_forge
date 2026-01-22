import asyncio
import base64
import json
import logging
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import yaml
import wandb
from wandb.apis.internal import Api
from wandb.sdk.launch.agent.agent import LaunchAgent
from wandb.sdk.launch.environment.abstract import AbstractEnvironment
from wandb.sdk.launch.registry.abstract import AbstractRegistry
from wandb.sdk.launch.registry.azure_container_registry import AzureContainerRegistry
from wandb.sdk.launch.registry.local_registry import LocalRegistry
from wandb.sdk.launch.runner.abstract import Status
from wandb.sdk.launch.runner.kubernetes_monitor import (
from wandb.util import get_module
from .._project_spec import EntryPoint, LaunchProject
from ..builder.build import get_env_vars_dict
from ..errors import LaunchError
from ..utils import (
from .abstract import AbstractRun, AbstractRunner
import kubernetes_asyncio  # type: ignore # noqa: E402
from kubernetes_asyncio import client  # noqa: E402
from kubernetes_asyncio.client.api.batch_v1_api import (  # type: ignore # noqa: E402
from kubernetes_asyncio.client.api.core_v1_api import (  # type: ignore # noqa: E402
from kubernetes_asyncio.client.api.custom_objects_api import (  # type: ignore # noqa: E402
from kubernetes_asyncio.client.models.v1_secret import (  # type: ignore # noqa: E402
from kubernetes_asyncio.client.rest import ApiException  # type: ignore # noqa: E402
def add_wandb_env(root: Union[dict, list], env_vars: Dict[str, str]) -> None:
    """Injects wandb environment variables into specs.

    Recursively walks the spec and injects the environment variables into
    every container spec. Containers are identified by the "containers" key.

    This function treats the WANDB_RUN_ID and WANDB_GROUP_ID environment variables
    specially. If they are present in the spec, they will be overwritten. If a setting
    for WANDB_RUN_ID is provided in env_vars, then that environment variable will only be
    set in the first container modified by this function.

    Arguments:
        root: The spec to modify.
        env_vars: The environment variables to inject.

    Returns: None.
    """
    for cont in yield_containers(root):
        env = cont.setdefault('env', [])
        env.extend([{'name': key, 'value': value} for key, value in env_vars.items()])
        cont['env'] = env
        if 'WANDB_RUN_ID' in env_vars:
            env_vars.pop('WANDB_RUN_ID')