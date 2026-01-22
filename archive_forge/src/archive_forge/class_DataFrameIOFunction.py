from __future__ import annotations
import json
from typing import Protocol, runtime_checkable
from uuid import uuid4
import fsspec
import pandas as pd
from fsspec.implementations.local import LocalFileSystem
from packaging.version import parse as parse_version
@runtime_checkable
class DataFrameIOFunction(Protocol):
    """DataFrame IO function with projectable columns

    Enables column projection in ``DataFrameIOLayer``.
    """

    @property
    def columns(self):
        """Return the current column projection"""
        raise NotImplementedError

    def project_columns(self, columns):
        """Return a new DataFrameIOFunction object
        with a new column projection
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Return a new DataFrame partition"""
        raise NotImplementedError