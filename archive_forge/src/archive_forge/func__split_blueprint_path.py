from __future__ import annotations
import importlib.util
import os
import sys
import typing as t
from datetime import datetime
from functools import lru_cache
from functools import update_wrapper
import werkzeug.utils
from werkzeug.exceptions import abort as _wz_abort
from werkzeug.utils import redirect as _wz_redirect
from werkzeug.wrappers import Response as BaseResponse
from .globals import _cv_request
from .globals import current_app
from .globals import request
from .globals import request_ctx
from .globals import session
from .signals import message_flashed
@lru_cache(maxsize=None)
def _split_blueprint_path(name: str) -> list[str]:
    out: list[str] = [name]
    if '.' in name:
        out.extend(_split_blueprint_path(name.rpartition('.')[0]))
    return out