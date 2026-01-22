import ast
import asyncio
import inspect
from functools import wraps
class _AsyncIOProxy:
    """Proxy-object for an asyncio

    Any coroutine methods will be wrapped in event_loop.run_
    """

    def __init__(self, obj, event_loop):
        self._obj = obj
        self._event_loop = event_loop

    def __repr__(self):
        return f'<_AsyncIOProxy({self._obj!r})>'

    def __getattr__(self, key):
        attr = getattr(self._obj, key)
        if inspect.iscoroutinefunction(attr):

            @wraps(attr)
            def _wrapped(*args, **kwargs):
                concurrent_future = asyncio.run_coroutine_threadsafe(attr(*args, **kwargs), self._event_loop)
                return asyncio.wrap_future(concurrent_future)
            return _wrapped
        else:
            return attr

    def __dir__(self):
        return dir(self._obj)