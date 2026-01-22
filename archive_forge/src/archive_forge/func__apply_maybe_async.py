from __future__ import annotations
import argparse
import collections
from functools import update_wrapper
import inspect
import itertools
import operator
import os
import re
import sys
from typing import TYPE_CHECKING
import uuid
import pytest
def _apply_maybe_async(obj, recurse=True):
    from sqlalchemy.testing import asyncio
    for name, value in vars(obj).items():
        if (callable(value) or isinstance(value, classmethod)) and (not getattr(value, '_maybe_async_applied', False)) and name.startswith('test_') and (not _is_wrapped_coroutine_function(value)):
            is_classmethod = False
            if isinstance(value, classmethod):
                value = value.__func__
                is_classmethod = True

            @_pytest_fn_decorator
            def make_async(fn, *args, **kwargs):
                return asyncio._maybe_async(fn, *args, **kwargs)
            do_async = make_async(value)
            if is_classmethod:
                do_async = classmethod(do_async)
            do_async._maybe_async_applied = True
            setattr(obj, name, do_async)
    if recurse:
        for cls in obj.mro()[1:]:
            if cls != object:
                _apply_maybe_async(cls, False)
    return obj