from collections import deque
from typing import Any, Callable, Hashable, Iterable, List, Optional
from queuelib.queue import BaseQueue
A round robin queue implemented using multiple internal queues (typically,
    FIFO queues). The internal queue must implement the following methods:
        * push(obj)
        * pop()
        * peek()
        * close()
        * __len__()
    The constructor receives a qfactory argument, which is a callable used to
    instantiate a new (internal) queue when a new key is allocated. The
    qfactory function is called with the key number as first and only argument.
    start_domains is a sequence of domains to initialize the queue with. If the
    queue was previously closed leaving some domain buckets non-empty, those
    domains should be passed in start_domains.

    The queue maintains a fifo queue of keys. The key that went last is popped
    first and the next queue for that key is then popped.
    