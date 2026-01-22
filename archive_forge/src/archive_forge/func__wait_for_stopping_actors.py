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
def _wait_for_stopping_actors(self, timeout=None):
    if len(self._actor_deletion_ids) == 0:
        return
    if timeout is not None:
        timeout = float(timeout)
    _, deleting = ray.wait(self._actor_deletion_ids, num_returns=len(self._actor_deletion_ids), timeout=timeout)
    self._actor_deletion_ids = deleting