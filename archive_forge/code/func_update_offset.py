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
def update_offset(self, offset: int) -> None:
    """Update the position offset assigned by the BarManager."""
    if offset != self.pos_offset:
        self.pos_offset = offset
        for bar in self.bars_by_uuid.values():
            bar.update_offset(offset)