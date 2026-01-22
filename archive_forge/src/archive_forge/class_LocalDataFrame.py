import json
from abc import abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union
import pandas as pd
import pyarrow as pa
from triad import SerializableRLock
from triad.collections.schema import Schema
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_or_throw
from triad.utils.pandas_like import PD_UTILS
from triad.utils.pyarrow import cast_pa_table
from .._utils.display import PrettyTable
from ..collections.yielded import Yielded
from ..dataset import (
from ..exceptions import FugueDataFrameOperationError
class LocalDataFrame(DataFrame):
    """Base class of all local dataframes. Please read
    :ref:`this <tutorial:tutorials/advanced/schema_dataframes:dataframe>`
    to understand the concept

    :param schema: a `schema-like <triad.collections.schema.Schema>`_ object

    .. note::

        This is an abstract class, and normally you don't construct it by yourself
        unless you are
        implementing a new :class:`~fugue.execution.execution_engine.ExecutionEngine`
    """

    def native_as_df(self) -> AnyDataFrame:
        return self.as_pandas()

    @property
    def is_local(self) -> bool:
        """Always True because it's a LocalDataFrame"""
        return True

    @property
    def num_partitions(self) -> int:
        """Always 1 because it's a LocalDataFrame"""
        return 1