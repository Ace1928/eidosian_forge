from types import SimpleNamespace
from typing import TYPE_CHECKING, Awaitable, Optional, Protocol, Type, TypeVar
import attr
from aiosignal import Signal
from multidict import CIMultiDict
from yarl import URL
from .client_reqrep import ClientResponse
def trace_config_ctx(self, trace_request_ctx: Optional[SimpleNamespace]=None) -> SimpleNamespace:
    """Return a new trace_config_ctx instance"""
    return self._trace_config_ctx_factory(trace_request_ctx=trace_request_ctx)