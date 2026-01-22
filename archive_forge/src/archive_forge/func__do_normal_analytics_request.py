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
def _do_normal_analytics_request(url: str, data: dict[str, Any]) -> None:
    data['ip_address'] = get_local_ip_address()
    try:
        httpx.post(url, data=data, timeout=5)
    except (httpx.ConnectError, httpx.ReadTimeout):
        pass