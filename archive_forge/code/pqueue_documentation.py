from typing import Any, Callable, Iterable, List, Optional
from queuelib.queue import BaseQueue
A priority queue implemented using multiple internal queues (typically,
    FIFO queues). The internal queue must implement the following methods:

        * push(obj)
        * pop()
        * peek()
        * close()
        * __len__()

    The constructor receives a qfactory argument, which is a callable used to
    instantiate a new (internal) queue when a new priority is allocated. The
    qfactory function is called with the priority number as first and only
    argument.

    Only integer priorities should be used. Lower numbers are higher
    priorities.

    startprios is a sequence of priorities to start with. If the queue was
    previously closed leaving some priority buckets non-empty, those priorities
    should be passed in startprios.

    