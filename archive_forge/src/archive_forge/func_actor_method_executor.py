import dis
import hashlib
import importlib
import inspect
import json
import logging
import os
import threading
import time
import traceback
from collections import defaultdict, namedtuple
from typing import Optional, Callable
import ray
import ray._private.profiling as profiling
from ray import cloudpickle as pickle
from ray._private import ray_constants
from ray._private.inspect_util import (
from ray._private.ray_constants import KV_NAMESPACE_FUNCTION_TABLE
from ray._private.utils import (
from ray._private.serialization import pickle_dumps
from ray._raylet import (
def actor_method_executor(__ray_actor, *args, **kwargs):
    is_bound = is_class_method(method) or is_static_method(type(__ray_actor), method_name)
    if is_bound:
        return method(*args, **kwargs)
    else:
        return method(__ray_actor, *args, **kwargs)