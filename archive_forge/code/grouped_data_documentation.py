from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from ._internal.table_block import TableBlockAccessor
from ray.data._internal import sort
from ray.data._internal.compute import ComputeStrategy
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.logical.interfaces import LogicalPlan
from ray.data._internal.logical.operators.all_to_all_operator import Aggregate
from ray.data._internal.plan import AllToAllStage
from ray.data._internal.push_based_shuffle import PushBasedShufflePlan
from ray.data._internal.shuffle import ShuffleOp, SimpleShufflePlan
from ray.data._internal.sort import SortKey
from ray.data.aggregate import AggregateFn, Count, Max, Mean, Min, Std, Sum
from ray.data.aggregate._aggregate import _AggregateOnKeyBase
from ray.data.block import (
from ray.data.context import DataContext
from ray.data.dataset import DataBatch, Dataset
from ray.util.annotations import PublicAPI
Compute block boundaries based on the key(s)