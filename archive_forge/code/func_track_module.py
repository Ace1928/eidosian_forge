import os
from collections import defaultdict
from typing import Callable, Dict, Iterable, Sequence, Set, List
from .. import importcompletion
def track_module(self, path: str) -> None:
    """
            Begins tracking this if activated, or remembers to track later.
            """
    if self.activated:
        self._add_module(path)
    else:
        self._add_module_later(path)