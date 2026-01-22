import atexit
import inspect
import os
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
from .utils import experimental, is_gradio_available
from .utils._deprecation import _deprecate_method
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
def _get_global_app() -> WebhooksServer:
    global _global_app
    if _global_app is None:
        _global_app = WebhooksServer()
    return _global_app