import importlib
import os
import secrets
import sys
import warnings
from textwrap import dedent
from typing import Any, Optional
from packaging import version
from pandas.util._decorators import doc  # type: ignore[attr-defined]
from modin.config.pubsub import (
def use_range_partitioning_groupby() -> bool:
    """
    Determine whether range-partitioning implementation for groupby was enabled by a user.

    This is a temporary helper function that queries ``RangePartitioning`` and deprecated
    ``RangePartitioningGroupby`` config variables in order to determine whether to range-part
    impl for groupby is enabled. Eventially this function should be removed together with
    ``RangePartitioningGroupby`` variable.

    Returns
    -------
    bool
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        old_range_part_var = RangePartitioningGroupby.get()
    return RangePartitioning.get() or old_range_part_var