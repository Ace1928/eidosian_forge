import asyncio
import functools
from typing import AsyncGenerator, Generic, Iterator, Optional, TypeVar
import grpc
from grpc import aio
from google.api_core import exceptions, grpc_helpers
def with_call(self, call):
    """Supplies the call object separately to keep __init__ clean."""
    self._call = call
    return self