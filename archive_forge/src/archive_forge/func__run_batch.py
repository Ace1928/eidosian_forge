import collections
import copy
import gc
import itertools
import logging
import os
import queue
import sys
import threading
import time
from multiprocessing import TimeoutError
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Tuple
import ray
from ray._private.usage import usage_lib
from ray.util import log_once
def _run_batch(self, actor_index, func, batch):
    actor, count = self._actor_pool[actor_index]
    object_ref = actor.run_batch.remote(func, batch)
    count += 1
    assert self._maxtasksperchild == -1 or count <= self._maxtasksperchild
    if count == self._maxtasksperchild:
        self._stop_actor(actor)
        actor, count = self._new_actor_entry()
    self._actor_pool[actor_index] = (actor, count)
    return object_ref