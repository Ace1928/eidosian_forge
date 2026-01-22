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
def _do_analytics_request(url: str, data: dict[str, Any]) -> None:
    if wasm_utils.IS_WASM:
        asyncio.ensure_future(_do_wasm_analytics_request(url=url, data=data))
    else:
        threading.Thread(target=_do_normal_analytics_request, kwargs={'url': url, 'data': data}).start()