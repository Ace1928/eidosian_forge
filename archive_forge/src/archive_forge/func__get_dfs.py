import inspect
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from typing import (
from uuid import uuid4
from triad import ParamDict, Schema, SerializableRLock, assert_or_throw, to_uuid
from triad.collections.function_wrapper import AnnotatedParam
from triad.exceptions import InvalidOperationError
from triad.utils.convert import to_size
from fugue.bag import Bag, LocalBag
from fugue.collections.partition import (
from fugue.collections.sql import StructuredRawSQL, TempTableName
from fugue.collections.yielded import PhysicalYielded, Yielded
from fugue.column import (
from fugue.constants import _FUGUE_GLOBAL_CONF, FUGUE_SQL_DEFAULT_DIALECT
from fugue.dataframe import AnyDataFrame, DataFrame, DataFrames, fugue_annotated_param
from fugue.dataframe.array_dataframe import ArrayDataFrame
from fugue.dataframe.dataframe import LocalDataFrame
from fugue.dataframe.utils import deserialize_df, serialize_df
from fugue.exceptions import FugueWorkflowRuntimeError
def _get_dfs(self, rows: List[Dict[str, Any]]) -> DataFrames:
    tdfs: Dict[Any, DataFrame] = {}
    for row in rows:
        df = deserialize_df(row[_FUGUE_SERIALIZED_BLOB_COL])
        if df is not None:
            if self.named:
                tdfs[row[_FUGUE_SERIALIZED_BLOB_NAME_COL]] = df
            else:
                tdfs[row[_FUGUE_SERIALIZED_BLOB_NO_COL]] = df
    dfs: Dict[Any, DataFrame] = {}
    for k, schema in self.schemas.items():
        if k in tdfs:
            dfs[k] = tdfs[k]
        else:
            dfs[k] = ArrayDataFrame([], schema)
    return DataFrames(dfs) if self.named else DataFrames(list(dfs.values()))