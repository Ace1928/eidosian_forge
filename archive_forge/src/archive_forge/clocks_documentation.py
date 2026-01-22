from __future__ import annotations
from itertools import islice
from operator import itemgetter
from threading import Lock
from typing import Any
Sort heap of events.

        List of tuples containing at least two elements, representing
        an event, where the first element is the event's scalar clock value,
        and the second element is the id of the process (usually
        ``"hostname:pid"``): ``sh([(clock, processid, ...?), (...)])``

        The list must already be sorted, which is why we refer to it as a
        heap.

        The tuple will not be unpacked, so more than two elements can be
        present.

        Will return the latest event.
        