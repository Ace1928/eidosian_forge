import heapq
import inspect
import unittest
from pbr.version import VersionInfo
def _call_result_method_if_exists(self, result, methodname, *args):
    """Call a method on a TestResult that may exist."""
    method = getattr(result, methodname, None)
    if callable(method):
        method(*args)