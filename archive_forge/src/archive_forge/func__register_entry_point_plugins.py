from __future__ import annotations
import itertools
import logging
import random
import sys
from array import array
from packaging.version import parse as parse_version
from dask._compatibility import importlib_metadata
from dask.utils import Dispatch
def _register_entry_point_plugins():
    """Register sizeof implementations exposed by the entry_point mechanism."""
    for entry_point in importlib_metadata.entry_points(group='dask.sizeof'):
        registrar = entry_point.load()
        try:
            registrar(sizeof)
        except Exception:
            logger.exception(f'Failed to register sizeof entry point {entry_point.name}')