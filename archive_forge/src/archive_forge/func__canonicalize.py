from typing import List, Optional, Tuple
from ray.data._internal.compute import get_compute, is_task_compute
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.operators.actor_pool_map_operator import (
from ray.data._internal.execution.operators.base_physical_operator import (
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.task_pool_map_operator import (
from ray.data._internal.logical.interfaces import PhysicalPlan, Rule
from ray.data._internal.logical.operators.all_to_all_operator import (
from ray.data._internal.logical.operators.map_operator import AbstractUDFMap
from ray.data._internal.stats import StatsDict
from ray.data.context import DataContext
def _canonicalize(remote_args: dict) -> dict:
    """Returns canonical form of given remote args."""
    remote_args = remote_args.copy()
    if 'num_cpus' not in remote_args or remote_args['num_cpus'] is None:
        remote_args['num_cpus'] = 1
    if 'num_gpus' not in remote_args or remote_args['num_gpus'] is None:
        remote_args['num_gpus'] = 0
    resources = remote_args.get('resources', {})
    for k, v in list(resources.items()):
        if v is None or v == 0.0:
            del resources[k]
    remote_args['resources'] = resources
    return remote_args