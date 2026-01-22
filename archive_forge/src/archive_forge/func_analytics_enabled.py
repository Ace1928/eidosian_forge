from __future__ import annotations
import asyncio
import json
import os
import threading
import urllib.parse
import warnings
from typing import Any
import httpx
from packaging.version import Version
import gradio
from gradio import wasm_utils
from gradio.context import Context
from gradio.utils import get_package_version
def analytics_enabled() -> bool:
    """
    Returns: True if analytics are enabled, False otherwise.
    """
    return os.getenv('GRADIO_ANALYTICS_ENABLED', 'True') == 'True'