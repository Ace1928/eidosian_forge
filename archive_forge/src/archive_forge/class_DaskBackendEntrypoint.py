from __future__ import annotations
from collections.abc import Callable
from functools import lru_cache, wraps
from typing import TYPE_CHECKING, Generic, TypeVar
from dask import config
from dask._compatibility import importlib_metadata
from dask.utils import funcname
class DaskBackendEntrypoint:
    """Base Collection-Backend Entrypoint Class

    Most methods in this class correspond to collection-creation
    for a specific library backend. Once a collection is created,
    the existing data will be used to dispatch compute operations
    within individual tasks. The backend is responsible for
    ensuring that these data-directed dispatch functions are
    registered when ``__init__`` is called.
    """

    @classmethod
    def to_backend_dispatch(cls):
        """Return a dispatch function to move data to this backend"""
        raise NotImplementedError

    @staticmethod
    def to_backend(data):
        """Create a new collection with this backend"""
        raise NotImplementedError