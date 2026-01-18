import logging
import queue
import time
from typing import Any, Callable, List, Sequence
import uuid
A helper that executes a callback for items sent in a queue.

    Calls a blocking ``get()`` on the ``queue`` until it encounters
    :attr:`STOP`.

    Args:
        queue:
            A Queue instance, appropriate for crossing the concurrency boundary
            implemented by ``executor``. Items will be popped off (with a blocking
            ``get()``) until :attr:`STOP` is encountered.
        callback:
            A callback that can process items pulled off of the queue. Multiple items
            will be passed to the callback in batches.
        max_items:
            The maximum amount of items that will be passed to the callback at a time.
        max_latency:
            The maximum amount of time in seconds to wait for additional items before
            executing the callback.
    