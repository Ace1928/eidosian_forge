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
class GRPCIntegration(Integration):
    identifier = 'grpc'

    @staticmethod
    def setup_once() -> None:
        import grpc
        grpc.insecure_channel = _wrap_channel_sync(grpc.insecure_channel)
        grpc.secure_channel = _wrap_channel_sync(grpc.secure_channel)
        grpc.intercept_channel = _wrap_intercept_channel(grpc.intercept_channel)
        grpc.aio.insecure_channel = _wrap_channel_async(grpc.aio.insecure_channel)
        grpc.aio.secure_channel = _wrap_channel_async(grpc.aio.secure_channel)
        grpc.server = _wrap_sync_server(grpc.server)
        grpc.aio.server = _wrap_async_server(grpc.aio.server)