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
def _get_or_allocate_bar_group(self, state: ProgressBarState):
    ptuple = (state['ip'], state['pid'])
    if ptuple not in self.bar_groups:
        offset = sum((p.slots_required() for p in self.bar_groups.values()))
        self.bar_groups[ptuple] = _BarGroup(state['ip'], state['pid'], offset)
    return self.bar_groups[ptuple]