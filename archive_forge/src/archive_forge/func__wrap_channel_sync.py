from functools import wraps
import grpc
from grpc import Channel, Server, intercept_channel
from grpc.aio import Channel as AsyncChannel
from grpc.aio import Server as AsyncServer
from sentry_sdk.integrations import Integration
from sentry_sdk._types import TYPE_CHECKING
from .client import ClientInterceptor
from .server import ServerInterceptor
from .aio.server import ServerInterceptor as AsyncServerInterceptor
from .aio.client import (
from .aio.client import (
from typing import Any, Optional, Sequence
def _wrap_channel_sync(func: Callable[P, Channel]) -> Callable[P, Channel]:
    """Wrapper for synchronous secure and insecure channel."""

    @wraps(func)
    def patched_channel(*args: Any, **kwargs: Any) -> Channel:
        channel = func(*args, **kwargs)
        if not ClientInterceptor._is_intercepted:
            ClientInterceptor._is_intercepted = True
            return intercept_channel(channel, ClientInterceptor())
        else:
            return channel
    return patched_channel