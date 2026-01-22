from collections import deque
from typing import Deque
class InstrumentationWarning(UserWarning):
    """Emitted when there's a problem with instrumenting a function for type checks."""

    def __init__(self, message: str):
        super().__init__(message)