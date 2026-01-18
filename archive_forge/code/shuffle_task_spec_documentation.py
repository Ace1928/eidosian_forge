import math
from typing import Callable, Iterable, List, Optional, Tuple, Union
import numpy as np
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.planner.exchange.interfaces import ExchangeTaskSpec
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata
from ray.data.context import MAX_SAFE_BLOCK_SIZE_FACTOR

    The implementation for shuffle tasks.

    This is used by random_shuffle() and repartition().
    