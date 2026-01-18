from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc

        a1  b1  c1  d1  e1
        /|\ /|\ /|\ /|  /
       / | X | X | X | /
      /  |/ \|/ \|/ \|/
    a0  b0  c0  d0  e0
    