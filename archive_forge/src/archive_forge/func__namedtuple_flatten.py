import dataclasses
import importlib
import json
import threading
import warnings
from collections import defaultdict, deque, namedtuple, OrderedDict
from typing import (
def _namedtuple_flatten(d: NamedTuple) -> Tuple[List[Any], Context]:
    return (list(d), type(d))