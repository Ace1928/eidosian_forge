from __future__ import annotations
import contextlib
from typing import Any, TypeVar, Callable, Awaitable, Iterator
from asyncpg.cursor import BaseCursor  # type: ignore
from sentry_sdk import Hub
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.tracing import Span
from sentry_sdk.tracing_utils import add_query_source, record_sql_queries
from sentry_sdk.utils import parse_version, capture_internal_exceptions
def _wrap_execute(f: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:

    async def _inner(*args: Any, **kwargs: Any) -> T:
        hub = Hub.current
        integration = hub.get_integration(AsyncPGIntegration)
        if integration is None or len(args) > 2:
            return await f(*args, **kwargs)
        query = args[1]
        with record_sql_queries(hub, None, query, None, None, executemany=False) as span:
            res = await f(*args, **kwargs)
        with capture_internal_exceptions():
            add_query_source(hub, span)
        return res
    return _inner