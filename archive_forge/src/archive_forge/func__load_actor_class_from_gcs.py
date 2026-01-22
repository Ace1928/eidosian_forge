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
def _load_actor_class_from_gcs(self, job_id, actor_creation_function_descriptor):
    """Load actor class from GCS."""
    key = make_function_table_key(b'ActorClass', job_id, actor_creation_function_descriptor.function_id.binary())
    vals = self._worker.gcs_client.internal_kv_get(key, KV_NAMESPACE_FUNCTION_TABLE)
    fields = ['job_id', 'class_name', 'module', 'class', 'actor_method_names']
    if vals is None:
        vals = {}
    else:
        vals = pickle.loads(vals)
    job_id_str, class_name, module, pickled_class, actor_method_names = (vals.get(field) for field in fields)
    class_name = ensure_str(class_name)
    module_name = ensure_str(module)
    job_id = ray.JobID(job_id_str)
    actor_method_names = json.loads(ensure_str(actor_method_names))
    actor_class = None
    try:
        with self.lock:
            actor_class = pickle.loads(pickled_class)
    except Exception:
        logger.debug('Failed to load actor class %s.', class_name)
        traceback_str = format_error_message(traceback.format_exc())
        actor_class = self._create_fake_actor_class(class_name, actor_method_names, traceback_str)
    actor_class.__module__ = module_name
    return actor_class