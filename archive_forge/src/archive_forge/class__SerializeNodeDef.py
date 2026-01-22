import dataclasses
import importlib
import json
import threading
import warnings
from collections import defaultdict, deque, namedtuple, OrderedDict
from typing import (
class _SerializeNodeDef(NamedTuple):
    typ: Type[Any]
    serialized_type_name: str
    to_dumpable_context: Optional[ToDumpableContextFn]
    from_dumpable_context: Optional[FromDumpableContextFn]