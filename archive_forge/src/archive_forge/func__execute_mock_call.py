import asyncio
import contextlib
import io
import inspect
import pprint
import sys
import builtins
import pkgutil
from asyncio import iscoroutinefunction
from types import CodeType, ModuleType, MethodType
from unittest.util import safe_repr
from functools import wraps, partial
from threading import RLock
def _execute_mock_call(self, /, *args, **kwargs):
    effect = self.side_effect
    if effect is not None:
        if _is_exception(effect):
            raise effect
        elif not _callable(effect):
            result = next(effect)
            if _is_exception(result):
                raise result
        else:
            result = effect(*args, **kwargs)
        if result is not DEFAULT:
            return result
    if self._mock_return_value is not DEFAULT:
        return self.return_value
    if self._mock_wraps is not None:
        return self._mock_wraps(*args, **kwargs)
    return self.return_value