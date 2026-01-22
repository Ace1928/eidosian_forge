from enum import Enum
from typing import Union, Callable, Optional, cast
class LazyList:

    def __init__(self, get_more, value_dict=None):
        self._data = []
        self._last_key = None
        self._exhausted = False
        self._all_loaded = False
        self._get_more = get_more
        self._value_dict = value_dict or {}

    def __iter__(self):
        if not self._all_loaded:
            self._load_all()
        data = self._data
        yield from data

    def __getitem__(self, index):
        if index >= len(self._data) and (not self._all_loaded):
            self._load_all()
        return self._data[index]

    def __len__(self):
        self._load_all()
        return len(self._data)

    def __repr__(self):
        self._load_all()
        repr_string = ', '.join([repr(item) for item in self._data])
        repr_string = '[%s]' % repr_string
        return repr_string

    def _load_all(self):
        while not self._exhausted:
            newdata, self._last_key, self._exhausted = self._get_more(last_key=self._last_key, value_dict=self._value_dict)
            self._data.extend(newdata)
        self._all_loaded = True