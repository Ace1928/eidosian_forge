import logging
import os
import time
from collections import namedtuple
from numbers import Number
from typing import Any, Dict, Optional
import ray
from ray._private.resource_spec import NODE_ID_PREFIX
def _get_used_avail_resources(self, total_allocated_resources: Dict[str, Any]):
    total_allocated_resources = total_allocated_resources.copy()
    used_cpu = total_allocated_resources.pop('CPU', 0)
    total_cpu = self._avail_resources.cpu
    used_gpu = total_allocated_resources.pop('GPU', 0)
    total_gpu = self._avail_resources.gpu
    custom_used_total = {name: (total_allocated_resources.get(name, 0.0), self._avail_resources.get_res_total(name)) for name in self._avail_resources.custom_resources if not name.startswith(NODE_ID_PREFIX) and (total_allocated_resources.get(name, 0.0) > 0 or '_group_' not in name)}
    return (used_cpu, total_cpu, used_gpu, total_gpu, custom_used_total)