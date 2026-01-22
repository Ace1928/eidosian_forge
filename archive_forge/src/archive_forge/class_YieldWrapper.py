import sys
from functools import wraps
from types import coroutine
import inspect
from inspect import (
import collections.abc
class YieldWrapper:

    def __init__(self, payload):
        self.payload = payload