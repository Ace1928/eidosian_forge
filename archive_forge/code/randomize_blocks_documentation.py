import copy
from collections import deque
from ray.data._internal.logical.interfaces import LogicalOperator, LogicalPlan, Rule
from ray.data._internal.logical.operators.all_to_all_operator import (
Rule for reordering RandomizeBlocks logical operator.

    Reordering RandomizeBlocks operators is to help fuse multiple
    AbstractUDFMap operators together for better performance.

    1. Dedupes multiple RandomizeBlocks operators if they are not seeded.
    2. Moves RandomizeBlocks operator to the end of a sequence of AbstractUDFMap
    operators. RandomizeBlocks operators are not moved across AbstractAllToAll operator
    boundaries.
    