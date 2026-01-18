import unittest
from contextlib import contextmanager
from llvmlite import ir
from numba.core import types, typing, callconv, cpu, cgutils
from numba.core.registry import cpu_target
def times3(i):
    return i * 3