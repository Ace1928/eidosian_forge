import asyncio
import logging
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
import kubernetes_asyncio  # type: ignore # noqa: F401
import urllib3
from kubernetes_asyncio import watch
from kubernetes_asyncio.client import (  # type: ignore  # noqa: F401
import wandb
from wandb.sdk.launch.agent import LaunchAgent
from wandb.sdk.launch.errors import LaunchError
from wandb.sdk.launch.runner.abstract import State, Status
from wandb.sdk.launch.utils import get_kube_context_and_api_client
class CustomResource:
    """Class for custom resources."""

    def __init__(self, group: str, version: str, plural: str) -> None:
        """Initialize the CustomResource."""
        self.group = group
        self.version = version
        self.plural = plural

    def __str__(self) -> str:
        """Return a string representation of the CustomResource."""
        return f'{self.group}/{self.version}/{self.plural}'

    def __hash__(self) -> int:
        """Return a hash of the CustomResource."""
        return hash(str(self))