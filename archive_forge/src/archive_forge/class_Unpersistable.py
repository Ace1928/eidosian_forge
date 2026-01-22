import json
from itertools import count
from typing import Any, Callable, Optional
from twisted.trial import unittest
from .._flatten import KeyFlattener, aFormatter, extractField, flattenEvent
from .._format import formatEvent
from .._interfaces import LogEvent
class Unpersistable:
    """
            Unpersitable object.
            """
    destructed = False

    def selfDestruct(self) -> None:
        """
                Self destruct.
                """
        self.destructed = True

    def __repr__(self) -> str:
        if self.destructed:
            return 'post-serialization garbage'
        else:
            return 'un-persistable'