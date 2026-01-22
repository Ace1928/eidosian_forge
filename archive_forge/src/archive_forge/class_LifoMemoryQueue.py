import glob
import json
import os
import sqlite3
import struct
from abc import abstractmethod
from collections import deque
from contextlib import suppress
from typing import Any, Optional
class LifoMemoryQueue(FifoMemoryQueue):
    """In-memory LIFO queue, API compliant with LifoDiskQueue."""

    def pop(self) -> Optional[Any]:
        return self.q.pop() if self.q else None

    def peek(self) -> Optional[Any]:
        return self.q[-1] if self.q else None