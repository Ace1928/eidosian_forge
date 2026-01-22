import abc
import logging
import os
import random
import shutil
import time
import urllib
import uuid
from collections import namedtuple
from typing import IO, List, Optional, Tuple
import ray
from ray._private.ray_constants import DEFAULT_OBJECT_PREFIX
from ray._raylet import ObjectRef
def _get_objects_from_store(self, object_refs):
    worker = ray._private.worker.global_worker
    ray_object_pairs = worker.core_worker.get_if_local(object_refs)
    return ray_object_pairs