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
class WorkerStandardStreamDispatcher:

    def __init__(self):
        self.handlers = []
        self._lock = threading.Lock()

    def add_handler(self, name: str, handler: Callable) -> None:
        with self._lock:
            self.handlers.append((name, handler))

    def remove_handler(self, name: str) -> None:
        with self._lock:
            new_handlers = [pair for pair in self.handlers if pair[0] != name]
            self.handlers = new_handlers

    def emit(self, data):
        with self._lock:
            for pair in self.handlers:
                _, handle = pair
                handle(data)