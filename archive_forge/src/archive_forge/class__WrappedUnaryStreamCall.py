import asyncio
import functools
from typing import AsyncGenerator, Generic, Iterator, Optional, TypeVar
import grpc
from grpc import aio
from google.api_core import exceptions, grpc_helpers
class _WrappedUnaryStreamCall(_WrappedStreamResponseMixin[P], aio.UnaryStreamCall):
    """Wrapped UnaryStreamCall to map exceptions."""