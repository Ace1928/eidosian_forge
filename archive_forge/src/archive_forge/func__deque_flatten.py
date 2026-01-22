import dataclasses
import importlib
import json
import threading
import warnings
from collections import defaultdict, deque, namedtuple, OrderedDict
from typing import (
def _deque_flatten(deq: Deque[Any]) -> Tuple[List[Any], Context]:
    return (list(deq), deq.maxlen)