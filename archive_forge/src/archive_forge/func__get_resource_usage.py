import logging
import time
from collections import Counter
from functools import reduce
from typing import Dict, List
from ray._private.gcs_utils import PlacementGroupTableData
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.util import (
from ray.core.generated.common_pb2 import PlacementStrategy
def _get_resource_usage(self):
    resources_used = {}
    resources_total = {}
    for ip, max_resources in self.static_resources_by_ip.items():
        avail_resources = self.dynamic_resources_by_ip[ip]
        for resource_id, amount in max_resources.items():
            used = amount - avail_resources[resource_id]
            if resource_id not in resources_used:
                resources_used[resource_id] = 0.0
                resources_total[resource_id] = 0.0
            resources_used[resource_id] += used
            resources_total[resource_id] += amount
            used = max(0, used)
    return (resources_used, resources_total)