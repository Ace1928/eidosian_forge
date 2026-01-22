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
def _predict_resolve(self, *data) -> Any:
    """Needed for gradio.load(), which has a slightly different signature for serializing/deserializing"""
    outputs = self.make_predict()(*data)
    if len(self.dependency['outputs']) == 1:
        return outputs[0]
    return outputs