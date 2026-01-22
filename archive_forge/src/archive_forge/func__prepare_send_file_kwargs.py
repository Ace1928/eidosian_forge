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
def _prepare_send_file_kwargs(**kwargs: t.Any) -> dict[str, t.Any]:
    if kwargs.get('max_age') is None:
        kwargs['max_age'] = current_app.get_send_file_max_age
    kwargs.update(environ=request.environ, use_x_sendfile=current_app.config['USE_X_SENDFILE'], response_class=current_app.response_class, _root_path=current_app.root_path)
    return kwargs