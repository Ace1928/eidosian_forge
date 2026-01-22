import asyncio
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union, cast
from jupyter_core.utils import ensure_async
from tornado.log import app_log
from tornado.web import HTTPError
from .utils import HTTP_METHOD_TO_AUTH_ACTION
def allow_unauthenticated(method: FuncT) -> FuncT:
    """A decorator for tornado.web.RequestHandler methods
    that allows any user to make the following request.

    Selectively disables the 'authentication' layer of REST API which
    is active when `ServerApp.allow_unauthenticated_access = False`.

    To be used exclusively on endpoints which may be considered public,
    for example the login page handler.

    .. versionadded:: 2.13

    Parameters
    ----------
    method : bound callable
        the endpoint method to remove authentication from.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        return method(self, *args, **kwargs)
    setattr(wrapper, '__allow_unauthenticated', True)
    return cast(FuncT, wrapper)