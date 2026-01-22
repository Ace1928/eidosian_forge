import asyncio
import contextlib
import contextvars
import threading
from typing import Any, Dict, Union
class _CVar:
    """Storage utility for Local."""

    def __init__(self) -> None:
        self._data: 'contextvars.ContextVar[Dict[str, Any]]' = contextvars.ContextVar('asgiref.local')

    def __getattr__(self, key):
        storage_object = self._data.get({})
        try:
            return storage_object[key]
        except KeyError:
            raise AttributeError(f'{self!r} object has no attribute {key!r}')

    def __setattr__(self, key: str, value: Any) -> None:
        if key == '_data':
            return super().__setattr__(key, value)
        storage_object = self._data.get({})
        storage_object[key] = value
        self._data.set(storage_object)

    def __delattr__(self, key: str) -> None:
        storage_object = self._data.get({})
        if key in storage_object:
            del storage_object[key]
            self._data.set(storage_object)
        else:
            raise AttributeError(f'{self!r} object has no attribute {key!r}')