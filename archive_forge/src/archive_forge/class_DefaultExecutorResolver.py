import asyncio
import concurrent.futures
import errno
import os
import sys
import socket
import ssl
import stat
from tornado.concurrent import dummy_executor, run_on_executor
from tornado.ioloop import IOLoop
from tornado.util import Configurable, errno_from_exception
from typing import List, Callable, Any, Type, Dict, Union, Tuple, Awaitable, Optional
class DefaultExecutorResolver(Resolver):
    """Resolver implementation using `.IOLoop.run_in_executor`.

    .. versionadded:: 5.0

    .. deprecated:: 6.2

       Use `DefaultLoopResolver` instead.
    """

    async def resolve(self, host: str, port: int, family: socket.AddressFamily=socket.AF_UNSPEC) -> List[Tuple[int, Any]]:
        result = await IOLoop.current().run_in_executor(None, _resolve_addr, host, port, family)
        return result