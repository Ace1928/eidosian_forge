import dataclasses
import importlib
import json
import threading
import warnings
from collections import defaultdict, deque, namedtuple, OrderedDict
from typing import (
Get a flat list of arguments to this function

    A slightly faster version of tree_leaves((args, kwargs))
    