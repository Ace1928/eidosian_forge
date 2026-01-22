import os
import sys
import _yappi
import pickle
import threading
import warnings
import types
import inspect
import itertools
from contextlib import contextmanager
class YStatsIndexable(YStats):

    def __init__(self):
        super().__init__()
        self._additional_indexing = {}

    def clear(self):
        super().clear()
        self._additional_indexing.clear()

    def pop(self):
        item = super().pop()
        self._additional_indexing.pop(item.index, None)
        self._additional_indexing.pop(item.full_name, None)
        return item

    def append(self, item):
        super().append(item)
        self._additional_indexing.setdefault(item.index, item)
        self._additional_indexing.setdefault(item.full_name, item)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._additional_indexing.get(key, None)
        elif isinstance(key, str):
            return self._additional_indexing.get(key, None)
        elif isinstance(key, YFuncStat) or isinstance(key, YChildFuncStat):
            return self._additional_indexing.get(key.index, None)
        return super().__getitem__(key)