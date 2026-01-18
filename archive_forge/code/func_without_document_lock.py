from __future__ import annotations
import logging # isort:skip
import asyncio
from functools import wraps
from typing import (
def without_document_lock(func: F) -> NoLockCallback[F]:
    """ Wrap a callback function to execute without first obtaining the
    document lock.

    Args:
        func (callable) : The function to wrap

    Returns:
        callable : a function wrapped to execute without a |Document| lock.

    While inside an unlocked callback, it is completely *unsafe* to modify
    ``curdoc()``. The value of ``curdoc()`` inside the callback will be a
    specially wrapped version of |Document| that only allows safe operations,
    which are:

    * :func:`~bokeh.document.Document.add_next_tick_callback`
    * :func:`~bokeh.document.Document.remove_next_tick_callback`

    Only these may be used safely without taking the document lock. To make
    other changes to the document, you must add a next tick callback and make
    your changes to ``curdoc()`` from that second callback.

    Attempts to otherwise access or change the Document will result in an
    exception being raised.

    ``func`` can be a synchronous function, an async function, or a function
    decorated with ``asyncio.coroutine``. The returned function will be an
    async function if ``func`` is any of the latter two.

    """
    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def _wrapper(*args: Any, **kw: Any) -> None:
            await func(*args, **kw)
    else:

        @wraps(func)
        def _wrapper(*args: Any, **kw: Any) -> None:
            func(*args, **kw)
    wrapper = cast(NoLockCallback[F], _wrapper)
    wrapper.nolock = True
    return wrapper