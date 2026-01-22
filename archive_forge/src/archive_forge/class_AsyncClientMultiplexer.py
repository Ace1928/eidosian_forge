import asyncio
import threading
from typing import Generic, TypeVar, Callable, Dict, Awaitable
class AsyncClientMultiplexer(Generic[_Key, _Client]):
    _OpenedClientFactory = Callable[[_Key], Awaitable[_Client]]
    _ClientCloser = Callable[[_Client], Awaitable[None]]
    _factory: _OpenedClientFactory
    _closer: _ClientCloser
    _live_clients: Dict[_Key, Awaitable[_Client]]

    def __init__(self, factory: _OpenedClientFactory, closer: _ClientCloser=lambda client: client.__aexit__(None, None, None)):
        self._factory = factory
        self._closer = closer
        self._live_clients = {}

    async def get_or_create(self, key: _Key) -> _Client:
        if key not in self._live_clients:
            self._live_clients[key] = asyncio.ensure_future(self._factory(key))
        future = self._live_clients[key]
        try:
            return await future
        except BaseException as e:
            if key in self._live_clients and self._live_clients[key] is future:
                del self._live_clients[key]
            raise e

    async def try_erase(self, key: _Key, client: _Client):
        if key not in self._live_clients:
            return
        client_future = self._live_clients[key]
        current_client = await client_future
        if current_client is not client:
            return
        if key not in self._live_clients or self._live_clients[key] is not client_future:
            return
        del self._live_clients[key]
        await self._closer(client)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        live_clients: Dict[_Key, Awaitable[_Client]]
        live_clients = self._live_clients
        self._live_clients = {}
        for topic, client in live_clients.items():
            await self._closer(await client)