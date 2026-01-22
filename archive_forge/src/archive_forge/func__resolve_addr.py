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
def _resolve_addr(host: str, port: int, family: socket.AddressFamily=socket.AF_UNSPEC) -> List[Tuple[int, Any]]:
    addrinfo = socket.getaddrinfo(host, port, family, socket.SOCK_STREAM)
    results = []
    for fam, socktype, proto, canonname, address in addrinfo:
        results.append((fam, address))
    return results