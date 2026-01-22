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
class OverrideResolver(Resolver):
    """Wraps a resolver with a mapping of overrides.

    This can be used to make local DNS changes (e.g. for testing)
    without modifying system-wide settings.

    The mapping can be in three formats::

        {
            # Hostname to host or ip
            "example.com": "127.0.1.1",

            # Host+port to host+port
            ("login.example.com", 443): ("localhost", 1443),

            # Host+port+address family to host+port
            ("login.example.com", 443, socket.AF_INET6): ("::1", 1443),
        }

    .. versionchanged:: 5.0
       Added support for host-port-family triplets.
    """

    def initialize(self, resolver: Resolver, mapping: dict) -> None:
        self.resolver = resolver
        self.mapping = mapping

    def close(self) -> None:
        self.resolver.close()

    def resolve(self, host: str, port: int, family: socket.AddressFamily=socket.AF_UNSPEC) -> Awaitable[List[Tuple[int, Any]]]:
        if (host, port, family) in self.mapping:
            host, port = self.mapping[host, port, family]
        elif (host, port) in self.mapping:
            host, port = self.mapping[host, port]
        elif host in self.mapping:
            host = self.mapping[host]
        return self.resolver.resolve(host, port, family)