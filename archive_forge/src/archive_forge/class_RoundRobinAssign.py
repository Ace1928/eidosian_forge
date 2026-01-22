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
class RoundRobinAssign:

    def __init__(self, locs):
        self.locs = locs
        self.i = 0

    def __call__(self, args):
        args = copy.deepcopy(args)
        args['scheduling_strategy'] = NodeAffinitySchedulingStrategy(self.locs[self.i], soft=True, _spill_on_unavailable=True)
        self.i += 1
        self.i %= len(self.locs)
        return args