import codecs
import functools
import importlib
import inspect
import json
import os
import re
import sys
import types
import warnings
from pathlib import Path
from textwrap import dedent, indent
from typing import (
import numpy as np
import pandas
from packaging import version
from pandas._typing import JSONSerializable
from pandas.util._decorators import Appender  # type: ignore
from pandas.util._print_versions import (  # type: ignore[attr-defined]
from modin._version import get_versions
from modin.config import DocModule, Engine, StorageFormat
def _inherit_docstrings(parent: object, excluded: List[object]=[], overwrite_existing: bool=False, apilink: Optional[Union[str, List[str]]]=None) -> Callable[[Fn], Fn]:
    """
    Create a decorator which overwrites decorated object docstring(s).

    It takes `parent` __doc__ attribute. Also overwrites __doc__ of
    methods and properties defined in the target or its ancestors if it's a class
    with the __doc__ of matching methods and properties from the `parent`.

    Parameters
    ----------
    parent : object
        Parent object from which the decorated object inherits __doc__.
    excluded : list, default: []
        List of parent objects from which the class does not
        inherit docstrings.
    overwrite_existing : bool, default: False
        Allow overwriting docstrings that already exist in
        the decorated class.
    apilink : str | List[str], optional
        If non-empty, insert the link(s) to pandas API documentation.
        Should be the prefix part in the URL template, e.g. "pandas.DataFrame".

    Returns
    -------
    callable
        Decorator which replaces the decorated object's documentation with `parent` documentation.

    Notes
    -----
    Keep in mind that the function will override docstrings even for attributes which
    are not defined in target class (but are defined in the ancestor class),
    which means that ancestor class attribute docstrings could also change.
    """
    imported_doc_module = importlib.import_module(DocModule.get())
    default_parent = parent
    if DocModule.get() != DocModule.default and 'pandas' in str(getattr(parent, '__module__', '')):
        parent = getattr(imported_doc_module, getattr(parent, '__name__', ''), parent)
    if parent != default_parent:
        apilink = None
        overwrite_existing = True

    def _documentable_obj(obj: object) -> bool:
        """Check if `obj` docstring could be patched."""
        return bool(callable(obj) or (isinstance(obj, property) and obj.fget) or (isinstance(obj, (staticmethod, classmethod)) and obj.__func__))

    def decorator(cls_or_func: Fn) -> Fn:
        if parent not in excluded:
            _replace_doc(parent, cls_or_func, overwrite_existing, apilink)
        if not isinstance(cls_or_func, types.FunctionType):
            seen = set()
            for base in cls_or_func.__mro__:
                if base is object:
                    continue
                for attr, obj in base.__dict__.items():
                    if attr in seen:
                        continue
                    seen.add(attr)
                    parent_obj = getattr(parent, attr, getattr(default_parent, attr, None))
                    if parent_obj in excluded or not _documentable_obj(parent_obj) or (not _documentable_obj(obj)):
                        continue
                    _replace_doc(parent_obj, obj, overwrite_existing, apilink, parent_cls=cls_or_func, attr_name=attr)
        return cls_or_func
    return decorator