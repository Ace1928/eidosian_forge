from __future__ import annotations
import datetime
import json
import os
import pathlib
import traceback
import types
from collections import OrderedDict, defaultdict
from enum import Enum
from hashlib import sha1
from importlib import import_module
from inspect import getfullargspec
from pathlib import Path
from uuid import UUID
def _serialize_callable(o):
    if isinstance(o, types.BuiltinFunctionType):
        bound = None
    else:
        bound = getattr(o, '__self__', None)
    if bound is not None:
        try:
            bound = MontyEncoder().default(bound)
        except TypeError:
            raise TypeError('Only bound methods of classes or MSONable instances are supported.')
    return {'@module': o.__module__, '@callable': getattr(o, '__qualname__', o.__name__), '@bound': bound}