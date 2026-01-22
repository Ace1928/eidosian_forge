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
class SafeWatch:
    """Wrapper for the kubernetes watch class that can recover in more situations."""

    def __init__(self, watcher: watch.Watch) -> None:
        """Initialize the SafeWatch."""
        self._watcher = watcher
        self._last_seen_resource_version: Optional[str] = None
        self._stopped = False

    async def stream(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Stream the watcher.

        This method will automatically resume the stream if it breaks. It will
        also save the resource version so that the stream can be resumed from
        the last seen resource version.
        """
        while True:
            try:
                async for event in self._watcher.stream(func, *args, **kwargs, timeout_seconds=30):
                    if self._stopped:
                        break
                    object = event.get('object')
                    if isinstance(object, dict):
                        self._last_seen_resource_version = object.get('metadata', dict()).get('resourceVersion')
                    else:
                        self._last_seen_resource_version = object.metadata.resource_version
                    kwargs['resource_version'] = self._last_seen_resource_version
                    yield event
                if self._stopped:
                    break
            except urllib3.exceptions.ProtocolError as e:
                wandb.termwarn(f'Broken event stream: {e}, attempting to recover')
            except ApiException as e:
                if e.status == 410:
                    del kwargs['resource_version']
                    self._last_seen_resource_version = None
            except Exception as E:
                wandb.termerror(f'Unknown exception in event stream: {E}, attempting to recover')