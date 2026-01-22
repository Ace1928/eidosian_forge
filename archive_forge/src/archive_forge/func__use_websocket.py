from __future__ import annotations
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
import httpx
import huggingface_hub
import websockets
from packaging import version
from gradio_client import serializing, utils
from gradio_client.exceptions import SerializationSetupError
from gradio_client.utils import (
def _use_websocket(self, dependency: dict) -> bool:
    queue_enabled = self.client.config.get('enable_queue', False)
    queue_uses_websocket = version.parse(self.client.config.get('version', '2.0')) >= version.Version('3.2')
    dependency_uses_queue = dependency.get('queue', False) is not False
    return queue_enabled and queue_uses_websocket and dependency_uses_queue