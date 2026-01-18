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
def yield_containers(root: Any) -> Iterator[dict]:
    """Yield all container specs in a manifest.

    Recursively traverses the manifest and yields all container specs. Container
    specs are identified by the presence of a "containers" key in the value.
    """
    if isinstance(root, dict):
        for k, v in root.items():
            if k == 'containers':
                if isinstance(v, list):
                    yield from v
            elif isinstance(v, (dict, list)):
                yield from yield_containers(v)
    elif isinstance(root, list):
        for item in root:
            yield from yield_containers(item)