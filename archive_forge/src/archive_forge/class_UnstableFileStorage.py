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
class UnstableFileStorage(FileSystemStorage):
    """This class is for testing with writing failure."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._failure_rate = 0.1
        self._partial_failure_ratio = 0.2

    def spill_objects(self, object_refs, owner_addresses) -> List[str]:
        r = random.random() < self._failure_rate
        failed = r < self._failure_rate
        partial_failed = r < self._partial_failure_ratio
        if failed:
            raise IOError('Spilling object failed')
        elif partial_failed:
            i = random.choice(range(len(object_refs)))
            return super().spill_objects(object_refs[:i], owner_addresses)
        else:
            return super().spill_objects(object_refs, owner_addresses)