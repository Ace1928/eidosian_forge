import ast
import hashlib
import inspect
import os
import re
import warnings
from collections import deque
from contextlib import contextmanager
from functools import partial
from importlib import import_module
from pkgutil import iter_modules
from types import ModuleType
from typing import (
from w3lib.html import replace_entities
from scrapy.item import Item
from scrapy.utils.datatypes import LocalWeakReferencedCache
from scrapy.utils.deprecate import ScrapyDeprecationWarning
from scrapy.utils.python import flatten, to_unicode
def warn_on_generator_with_return_value(spider: 'Spider', callable: Callable) -> None:
    """
    Logs a warning if a callable is a generator function and includes
    a 'return' statement with a value different than None
    """
    try:
        if is_generator_with_return_value(callable):
            warnings.warn(f'The "{spider.__class__.__name__}.{callable.__name__}" method is a generator and includes a "return" statement with a value different than None. This could lead to unexpected behaviour. Please see https://docs.python.org/3/reference/simple_stmts.html#the-return-statement for details about the semantics of the "return" statement within generators', stacklevel=2)
    except IndentationError:
        callable_name = spider.__class__.__name__ + '.' + callable.__name__
        warnings.warn(f'Unable to determine whether or not "{callable_name}" is a generator with a return value. This will not prevent your code from working, but it prevents Scrapy from detecting potential issues in your implementation of "{callable_name}". Please, report this in the Scrapy issue tracker (https://github.com/scrapy/scrapy/issues), including the code of "{callable_name}"', stacklevel=2)