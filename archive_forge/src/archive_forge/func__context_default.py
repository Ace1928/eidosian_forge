from __future__ import annotations
import typing as t
import zmq.asyncio
from traitlets import Instance, Type
from ..channels import AsyncZMQSocketChannel, HBChannel
from ..client import KernelClient, reqrep
def _context_default(self) -> zmq.asyncio.Context:
    self._created_context = True
    return zmq.asyncio.Context()