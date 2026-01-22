import math
import threading
import time
from typing import Dict, List
import ray
from ray.data.context import DataContext
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def _aggregate_requests(self) -> List[Dict]:
    req = []
    for _, (r, _) in self._resource_requests.items():
        req.extend(r)

    def get_cpus(req):
        num_cpus = 0
        for r in req:
            if 'CPU' in r:
                num_cpus += r['CPU']
        return num_cpus
    num_cpus = get_cpus(req)
    if num_cpus > 0:
        total = ray.cluster_resources()
        if 'CPU' in total and num_cpus <= total['CPU']:
            delta = math.ceil(ARTIFICIAL_CPU_SCALING_FACTOR * total['CPU']) - num_cpus
            req.extend([{'CPU': 1}] * delta)
    return req