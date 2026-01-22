import functools
import inspect
import itertools
import logging
import sys
import threading
import types
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import (
class CallError(Error):
    """Call to callable object fails."""

    def __str__(self) -> str:
        if len(self.args) == 1:
            return self.args[0]
        instance, method, args, kwargs, original_error, stack = self.args
        cls = instance.__class__.__name__ if instance is not None else ''
        full_method = '.'.join((cls, method.__name__)).strip('.')
        parameters = ', '.join(itertools.chain((repr(arg) for arg in args), ('%s=%r' % (key, value) for key, value in kwargs.items())))
        return 'Call to %s(%s) failed: %s (injection stack: %r)' % (full_method, parameters, original_error, [level[0] for level in stack])