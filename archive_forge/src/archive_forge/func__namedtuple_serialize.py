import dataclasses
import importlib
import json
import threading
import warnings
from collections import defaultdict, deque, namedtuple, OrderedDict
from typing import (
def _namedtuple_serialize(context: Context) -> DumpableContext:
    json_namedtuple = {'class_name': context.__name__, 'fields': context._fields}
    return json_namedtuple