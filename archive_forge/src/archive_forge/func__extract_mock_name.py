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
def _extract_mock_name(self):
    _name_list = [self._mock_new_name]
    _parent = self._mock_new_parent
    last = self
    dot = '.'
    if _name_list == ['()']:
        dot = ''
    while _parent is not None:
        last = _parent
        _name_list.append(_parent._mock_new_name + dot)
        dot = '.'
        if _parent._mock_new_name == '()':
            dot = ''
        _parent = _parent._mock_new_parent
    _name_list = list(reversed(_name_list))
    _first = last._mock_name or 'mock'
    if len(_name_list) > 1:
        if _name_list[1] not in ('()', '().'):
            _first += '.'
    _name_list[0] = _first
    return ''.join(_name_list)