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
Make an executor that wraps a user-defined actor method.
        The wrapped method updates the worker's internal state and performs any
        necessary checkpointing operations.
        Args:
            method_name: The name of the actor method.
            method: The actor method to wrap. This should be a
                method defined on the actor class and should therefore take an
                instance of the actor as the first argument.
            actor_imported: Whether the actor has been imported.
                Checkpointing operations will not be run if this is set to
                False.
        Returns:
            A function that executes the given actor method on the worker's
                stored instance of the actor. The function also updates the
                worker's internal state to record the executed method.
        