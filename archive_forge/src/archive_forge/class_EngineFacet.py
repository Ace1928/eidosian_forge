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
class EngineFacet(FugueEngineBase):
    """The base class for different factes of the execution engines.

    :param execution_engine: the execution engine this sql engine will run on
    """

    def __init__(self, execution_engine: 'ExecutionEngine') -> None:
        tp = self.execution_engine_constraint
        if not isinstance(execution_engine, tp):
            raise TypeError(f'{self} expects the engine type to be {tp}, but got {type(execution_engine)}')
        self._execution_engine = execution_engine

    @property
    def execution_engine(self) -> 'ExecutionEngine':
        """the execution engine this sql engine will run on"""
        return self._execution_engine

    @property
    def log(self) -> logging.Logger:
        return self.execution_engine.log

    @property
    def conf(self) -> ParamDict:
        return self.execution_engine.conf

    def to_df(self, df: AnyDataFrame, schema: Any=None) -> DataFrame:
        return self.execution_engine.to_df(df, schema)

    @property
    def execution_engine_constraint(self) -> Type['ExecutionEngine']:
        """This defines the required ExecutionEngine type of this facet

        :return: a subtype of :class:`~.ExecutionEngine`
        """
        return ExecutionEngine