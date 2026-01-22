from __future__ import annotations
from abc import ABC, abstractmethod
class CustomIterator(ABC):
    """Lazy custom iteration and item access."""

    def __init__(self, obj):
        self.obj = obj
        self._iter = 0

    @abstractmethod
    def __getitem__(self, key):
        """Get next item"""
        pass

    def __repr__(self):
        return f'<{type(self.obj)}_iterator at {hex(id(self))}>'

    def __len__(self):
        return len(self.obj)

    def __iter__(self):
        self._iter = 0
        return self

    def __next__(self):
        if self._iter >= len(self):
            raise StopIteration
        self._iter += 1
        return self[self._iter - 1]