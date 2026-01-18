import builtins
import copy
import json
import logging
import os
import sys
import threading
import uuid
from typing import Any, Dict, Iterable, Optional
import colorama
import ray
from ray._private.ray_constants import env_bool
from ray.util.debug import log_once
def unhide_bars(self) -> None:
    """Opposite of hide_bars()."""
    with self.lock:
        if self.in_hidden_state:
            self.in_hidden_state = False
            for group in self.bar_groups.values():
                group.unhide_bars()