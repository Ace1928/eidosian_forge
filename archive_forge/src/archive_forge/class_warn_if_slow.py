import copy
import glob
import inspect
import logging
import os
import threading
import time
from collections import defaultdict
from datetime import datetime
from numbers import Number
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union
import numpy as np
import psutil
import ray
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.air._internal.json import SafeFallbackEncoder  # noqa
from ray.air._internal.util import (  # noqa: F401
from ray._private.dict import (  # noqa: F401
@DeveloperAPI
class warn_if_slow:
    """Prints a warning if a given operation is slower than 500ms.

    Example:
        >>> from ray.tune.utils.util import warn_if_slow
        >>> something = ... # doctest: +SKIP
        >>> with warn_if_slow("some_operation"): # doctest: +SKIP
        ...    ray.get(something) # doctest: +SKIP
    """
    DEFAULT_THRESHOLD = float(os.environ.get('TUNE_WARN_THRESHOLD_S', 0.5))
    DEFAULT_MESSAGE = 'The `{name}` operation took {duration:.3f} s, which may be a performance bottleneck.'

    def __init__(self, name: str, threshold: Optional[float]=None, message: Optional[str]=None, disable: bool=False):
        self.name = name
        self.threshold = threshold or self.DEFAULT_THRESHOLD
        self.message = message or self.DEFAULT_MESSAGE
        self.too_slow = False
        self.disable = disable

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        now = time.time()
        if self.disable:
            return
        if now - self.start > self.threshold and now - START_OF_TIME > 60.0:
            self.too_slow = True
            duration = now - self.start
            logger.warning(self.message.format(name=self.name, duration=duration))