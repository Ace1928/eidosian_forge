import copy
import functools
import itertools
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Set, Union
import ray
from ray import ObjectRef
from ray._raylet import ObjectRefGenerator
from ray.data._internal.compute import (
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.interfaces.physical_operator import (
from ray.data._internal.execution.operators.base_physical_operator import (
from ray.data._internal.execution.operators.map_transformer import (
from ray.data._internal.stats import StatsDict
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata
from ray.data.context import DataContext
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def _get_runtime_ray_remote_args(self, input_bundle: Optional[RefBundle]=None) -> Dict[str, Any]:
    ray_remote_args = copy.deepcopy(self._ray_remote_args)
    if 'scheduling_strategy' not in ray_remote_args:
        ctx = DataContext.get_current()
        if input_bundle and input_bundle.size_bytes() > ctx.large_args_threshold:
            ray_remote_args['scheduling_strategy'] = ctx.scheduling_strategy_large_args
            self._remote_args_for_metrics = copy.deepcopy(ray_remote_args)
        else:
            ray_remote_args['scheduling_strategy'] = ctx.scheduling_strategy
            if 'scheduling_strategy' not in self._remote_args_for_metrics:
                self._remote_args_for_metrics = copy.deepcopy(ray_remote_args)
    if self._ray_remote_args_factory:
        return self._ray_remote_args_factory(ray_remote_args)
    return ray_remote_args