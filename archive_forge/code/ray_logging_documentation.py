import colorama
from dataclasses import dataclass
import logging
import os
import re
import sys
import threading
import time
from typing import Callable, Dict, List, Set, Tuple, Any, Optional
import ray
from ray.experimental.tqdm_ray import RAY_TQDM_MAGIC
from ray._private.ray_constants import (
from ray.util.debug import log_once
Return all buffered log messages and clear the buffer.

        Returns:
            List of log batches to print.
        