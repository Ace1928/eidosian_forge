import dataclasses
import importlib
import json
import threading
import warnings
from collections import defaultdict, deque, namedtuple, OrderedDict
from typing import (
def _ordereddict_unflatten(values: Iterable[Any], context: Context) -> GenericOrderedDict[Any, Any]:
    return OrderedDict(((key, value) for key, value in zip(context, values)))