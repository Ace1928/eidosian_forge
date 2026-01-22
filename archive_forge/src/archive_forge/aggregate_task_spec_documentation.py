from typing import List, Optional, Tuple, Union
from ray.data._internal.planner.exchange.interfaces import ExchangeTaskSpec
from ray.data._internal.sort import SortKey
from ray.data._internal.table_block import TableBlockAccessor
from ray.data.aggregate import AggregateFn, Count
from ray.data.aggregate._aggregate import _AggregateOnKeyBase
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata, KeyType
Prune unused columns from block before aggregate.