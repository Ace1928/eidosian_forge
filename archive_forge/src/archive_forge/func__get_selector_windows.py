from __future__ import annotations
import asyncio
import selectors
import sys
import warnings
from asyncio import Future, SelectorEventLoop
from weakref import WeakKeyDictionary
import zmq as _zmq
from zmq import _future
def _get_selector_windows(asyncio_loop) -> asyncio.AbstractEventLoop:
    """Get selector-compatible loop

    Returns an object with ``add_reader`` family of methods,
    either the loop itself or a SelectorThread instance.

    Workaround Windows proactor removal of
    *reader methods, which we need for zmq sockets.
    """
    if asyncio_loop in _selectors:
        return _selectors[asyncio_loop]
    if hasattr(asyncio, 'ProactorEventLoop') and isinstance(asyncio_loop, asyncio.ProactorEventLoop):
        try:
            from tornado.platform.asyncio import AddThreadSelectorEventLoop
        except ImportError:
            raise RuntimeError("Proactor event loop does not implement add_reader family of methods required for zmq. zmq will work with proactor if tornado >= 6.1 can be found. Use `asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())` or install 'tornado>=6.1' to avoid this error.")
        warnings.warn('Proactor event loop does not implement add_reader family of methods required for zmq. Registering an additional selector thread for add_reader support via tornado. Use `asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())` to avoid this warning.', RuntimeWarning, stacklevel=5)
        selector_loop = _selectors[asyncio_loop] = AddThreadSelectorEventLoop(asyncio_loop)
        loop_close = asyncio_loop.close

        def _close_selector_and_loop():
            asyncio_loop.close = loop_close
            _selectors.pop(asyncio_loop, None)
            selector_loop.close()
        asyncio_loop.close = _close_selector_and_loop
        return selector_loop
    else:
        return asyncio_loop