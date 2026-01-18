from __future__ import annotations
import os
import sys
from array import array
import pytest
from dask.multiprocessing import get_context
from dask.sizeof import sizeof
from dask.utils import funcname
2+ contiguous columns of the same dtype in the same DataFrame share the same
    surface thus have lower overhead
    