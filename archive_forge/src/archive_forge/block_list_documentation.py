import math
from typing import Iterator, List, Optional, Tuple
import numpy as np
from ray.data._internal.memory_tracing import trace_allocation
from ray.data.block import Block, BlockMetadata
from ray.types import ObjectRef
Randomizes the order of the blocks.

        Args:
            seed: Fix the random seed to use, otherwise one will be chosen
                based on system randomness.
        