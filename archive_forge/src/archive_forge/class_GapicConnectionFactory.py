from typing import (
import asyncio
from google.api_core.exceptions import GoogleAPICallError, FailedPrecondition
from google.cloud.pubsublite.internal.wire.connection import (
from google.cloud.pubsublite.internal.wire.work_item import WorkItem
from google.cloud.pubsublite.internal.wire.permanent_failable import PermanentFailable
class GapicConnectionFactory(ConnectionFactory[Request, Response]):
    """A ConnectionFactory that produces GapicConnections."""
    _producer = Callable[[AsyncIterator[Request]], Awaitable[AsyncIterable[Response]]]

    def __init__(self, producer: Callable[[AsyncIterator[Request]], Awaitable[AsyncIterable[Response]]]):
        self._producer = producer

    async def new(self) -> Connection[Request, Response]:
        conn = GapicConnection[Request, Response]()
        response_fut = self._producer(conn)
        response_iterable = await response_fut
        conn.set_response_it(response_iterable.__aiter__())
        return conn