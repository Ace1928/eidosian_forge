import asyncio
import functools
from typing import AsyncGenerator, Generic, Iterator, Optional, TypeVar
import grpc
from grpc import aio
from google.api_core import exceptions, grpc_helpers
class _WrappedStreamStreamCall(_WrappedStreamRequestMixin, _WrappedStreamResponseMixin[P], aio.StreamStreamCall):
    """Wrapped StreamStreamCall to map exceptions."""