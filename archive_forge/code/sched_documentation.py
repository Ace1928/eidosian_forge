import time
import heapq
from collections import namedtuple
from itertools import count
import threading
from time import monotonic as _time
An ordered list of upcoming events.

        Events are named tuples with fields for:
            time, priority, action, arguments, kwargs

        