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
class MinPartitionSize(EnvironmentVariable, type=int):
    """
    Minimum number of rows/columns in a single pandas partition split.

    Once a partition for a pandas dataframe has more than this many elements,
    Modin adds another partition.
    """
    varname = 'MODIN_MIN_PARTITION_SIZE'
    default = 32

    @classmethod
    def put(cls, value: int) -> None:
        """
        Set ``MinPartitionSize`` with extra checks.

        Parameters
        ----------
        value : int
            Config value to set.
        """
        if value <= 0:
            raise ValueError(f'Min partition size should be > 0, passed value {value}')
        super().put(value)

    @classmethod
    def get(cls) -> int:
        """
        Get ``MinPartitionSize`` with extra checks.

        Returns
        -------
        int
        """
        min_partition_size = super().get()
        assert min_partition_size > 0, '`min_partition_size` should be > 0'
        return min_partition_size