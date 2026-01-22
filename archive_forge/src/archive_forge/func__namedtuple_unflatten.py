import dataclasses
import importlib
import json
import threading
import warnings
from collections import defaultdict, deque, namedtuple, OrderedDict
from typing import (
def _namedtuple_unflatten(values: Iterable[Any], context: Context) -> NamedTuple:
    return cast(NamedTuple, context(*values))