import dataclasses
import importlib
import json
import threading
import warnings
from collections import defaultdict, deque, namedtuple, OrderedDict
from typing import (
def _is_leaf(tree: PyTree) -> bool:
    return _get_node_type(tree) not in SUPPORTED_NODES