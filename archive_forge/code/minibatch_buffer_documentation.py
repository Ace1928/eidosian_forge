from typing import Any, Tuple
import queue
Get a new batch from the internal ring buffer.

        Returns:
           buf: Data item saved from inqueue.
           released: True if the item is now removed from the ring buffer.
        