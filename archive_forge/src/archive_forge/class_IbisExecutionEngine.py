import itertools
import logging
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type
import ibis
from ibis import BaseBackend
from triad import assert_or_throw
from fugue import StructuredRawSQL
from fugue.bag import Bag, LocalBag
from fugue.collections.partition import (
from fugue.dataframe import DataFrame, DataFrames, LocalDataFrame
from fugue.dataframe.utils import get_join_schemas
from fugue.execution.execution_engine import ExecutionEngine, MapEngine, SQLEngine
from ._compat import IbisTable
from ._utils import to_ibis_schema
from .dataframe import IbisDataFrame
class IbisExecutionEngine(ExecutionEngine):
    """The base execution engine using Ibis.
    Please read |ExecutionEngineTutorial| to understand this important Fugue concept

    :param conf: |ParamsLikeObject|, read |FugueConfig| to learn Fugue specific options
    """

    def __init__(self, conf: Any):
        super().__init__(conf)
        self._non_ibis_engine = self.create_non_ibis_execution_engine()

    @abstractmethod
    def create_non_ibis_execution_engine(self) -> ExecutionEngine:
        """Create the execution engine that handles operations beyond SQL"""
        raise NotImplementedError

    def is_non_ibis(self, ds: Any) -> bool:
        return not isinstance(ds, (IbisDataFrame, IbisTable))

    @property
    def non_ibis_engine(self) -> ExecutionEngine:
        return self._non_ibis_engine

    @property
    def ibis_sql_engine(self) -> IbisSQLEngine:
        return self.sql_engine

    def create_default_map_engine(self) -> MapEngine:
        return IbisMapEngine(self)

    @property
    def log(self) -> logging.Logger:
        return self.non_ibis_engine.log

    def get_current_parallelism(self) -> int:
        return self.non_ibis_engine.get_current_parallelism()

    def to_df(self, df: Any, schema: Any=None) -> DataFrame:
        if self.is_non_ibis(df):
            return self._to_non_ibis_dataframe(df, schema)
        return self._to_ibis_dataframe(df, schema=schema)

    def repartition(self, df: DataFrame, partition_spec: PartitionSpec) -> DataFrame:
        if self.is_non_ibis(df):
            return self.non_ibis_engine.repartition(df, partition_spec=partition_spec)
        self.log.warning("%s doesn't respect repartition", self)
        return df

    def broadcast(self, df: DataFrame) -> DataFrame:
        if self.is_non_ibis(df):
            return self.non_ibis_engine.broadcast(df)
        return df

    def persist(self, df: DataFrame, lazy: bool=False, **kwargs: Any) -> DataFrame:
        if self.is_non_ibis(df):
            return self.non_ibis_engine.persist(df, lazy=lazy, **kwargs)
        return self.ibis_sql_engine.persist(df, lazy=lazy, **kwargs)

    def join(self, df1: DataFrame, df2: DataFrame, how: str, on: Optional[List[str]]=None) -> DataFrame:
        if self.is_non_ibis(df1) and self.is_non_ibis(df2):
            return self.non_ibis_engine.join(df1, df2, how=how, on=on)
        return self.ibis_sql_engine.join(df1, df2, how=how, on=on)

    def union(self, df1: DataFrame, df2: DataFrame, distinct: bool=True) -> DataFrame:
        if self.is_non_ibis(df1) and self.is_non_ibis(df2):
            return self.non_ibis_engine.union(df1, df2, distinct=distinct)
        return self.ibis_sql_engine.union(df1, df2, distinct=distinct)

    def subtract(self, df1: DataFrame, df2: DataFrame, distinct: bool=True) -> DataFrame:
        if self.is_non_ibis(df1) and self.is_non_ibis(df2):
            return self.non_ibis_engine.subtract(df1, df2, distinct=distinct)
        return self.ibis_sql_engine.subtract(df1, df2, distinct=distinct)

    def intersect(self, df1: DataFrame, df2: DataFrame, distinct: bool=True) -> DataFrame:
        if self.is_non_ibis(df1) and self.is_non_ibis(df2):
            return self.non_ibis_engine.intersect(df1, df2, distinct=distinct)
        return self.ibis_sql_engine.intersect(df1, df2, distinct=distinct)

    def distinct(self, df: DataFrame) -> DataFrame:
        if self.is_non_ibis(df):
            return self.non_ibis_engine.distinct(df)
        return self.ibis_sql_engine.distinct(df)

    def dropna(self, df: DataFrame, how: str='any', thresh: int=None, subset: Optional[List[str]]=None) -> DataFrame:
        if self.is_non_ibis(df):
            return self.non_ibis_engine.dropna(df, how=how, thresh=thresh, subset=subset)
        return self.ibis_sql_engine.dropna(df, how=how, thresh=thresh, subset=subset)

    def fillna(self, df: DataFrame, value: Any, subset: List[str]=None) -> DataFrame:
        if self.is_non_ibis(df):
            return self.non_ibis_engine.fillna(df, value=value, subset=subset)
        return self.ibis_sql_engine.fillna(df, value=value, subset=subset)

    def sample(self, df: DataFrame, n: Optional[int]=None, frac: Optional[float]=None, replace: bool=False, seed: Optional[int]=None) -> DataFrame:
        if self.is_non_ibis(df):
            return self.non_ibis_engine.sample(df, n=n, frac=frac, replace=replace, seed=seed)
        return self.ibis_sql_engine.sample(df, n=n, frac=frac, replace=replace, seed=seed)

    def take(self, df: DataFrame, n: int, presort: str, na_position: str='last', partition_spec: Optional[PartitionSpec]=None) -> DataFrame:
        if self.is_non_ibis(df):
            return self.non_ibis_engine.take(df, n=n, presort=presort, na_position=na_position, partition_spec=partition_spec)
        return self.ibis_sql_engine.take(df, n=n, presort=presort, na_position=na_position, partition_spec=partition_spec)

    def _to_ibis_dataframe(self, df: Any, schema: Any=None) -> DataFrame:
        """Create ``IbisDataFrame`` from the dataframe like input

        :param df: dataframe like object
        :param schema: dataframe schema, defaults to None
        :return: the IbisDataFrame
        """
        return self.sql_engine.to_df(df, schema=schema)

    def _to_non_ibis_dataframe(self, df: Any, schema: Any=None) -> DataFrame:
        """Create ``DataFrame`` for map operations
        from the dataframe like input

        :param df: dataframe like object
        :param schema: dataframe schema, defaults to None
        :return: the DataFrame
        """
        return self.non_ibis_engine.to_df(df, schema)