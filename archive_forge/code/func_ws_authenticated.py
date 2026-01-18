import asyncio
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union, cast
from jupyter_core.utils import ensure_async
from tornado.log import app_log
from tornado.web import HTTPError
from .utils import HTTP_METHOD_TO_AUTH_ACTION
def ws_authenticated(method: FuncT) -> FuncT:
    """A decorator for websockets derived from `WebSocketHandler`
    that authenticates user before allowing to proceed.

    Differently from tornado.web.authenticated, does not redirect
    to the login page, which would be meaningless for websockets.

    .. versionadded:: 2.13

    Parameters
    ----------
    method : bound callable
        the endpoint method to add authentication for.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        user = self.current_user
        if user is None:
            self.log.warning("Couldn't authenticate WebSocket connection")
            raise HTTPError(403)
        return method(self, *args, **kwargs)
    setattr(wrapper, '__allow_unauthenticated', False)
    return cast(FuncT, wrapper)