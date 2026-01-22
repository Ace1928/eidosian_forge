from typing import Any, Callable, Dict, List, Optional, Type, Union
import pyarrow as pa
import ray
from duckdb import DuckDBPyConnection
from packaging import version
from triad import Schema, assert_or_throw, to_uuid
from triad.utils.threading import RunOnce
from fugue import (
from fugue.constants import KEYWORD_PARALLELISM, KEYWORD_ROWCOUNT
from fugue_duckdb.dataframe import DuckDataFrame
from fugue_duckdb.execution_engine import DuckExecutionEngine
from ._constants import FUGUE_RAY_DEFAULT_BATCH_SIZE, FUGUE_RAY_ZERO_COPY
from ._utils.cluster import get_default_partitions, get_default_shuffle_partitions
from ._utils.dataframe import add_coarse_partition_key, add_partition_key
from ._utils.io import RayIO
from .dataframe import RayDataFrame
class RayExecutionEngine(DuckExecutionEngine):
    """A hybrid engine of Ray and DuckDB as Phase 1 of Fugue Ray integration.
    Most operations will be done by DuckDB, but for ``map``, it will use Ray.

    :param conf: |ParamsLikeObject|, read |FugueConfig| to learn Fugue specific options
    :param connection: DuckDB connection
    """

    def __init__(self, conf: Any=None, connection: Optional[DuckDBPyConnection]=None):
        if not ray.is_initialized():
            ray.init()
        super().__init__(conf, connection)
        self._io = RayIO(self)

    def __repr__(self) -> str:
        return 'RayExecutionEngine'

    @property
    def is_distributed(self) -> bool:
        return True

    def create_default_map_engine(self) -> MapEngine:
        return RayMapEngine(self)

    def get_current_parallelism(self) -> int:
        res = ray.cluster_resources()
        n = res.get('CPU', 0)
        if n == 0:
            res.get('cpu', 0)
        return int(n)

    def to_df(self, df: Any, schema: Any=None) -> DataFrame:
        return self._to_ray_df(df, schema=schema)

    def repartition(self, df: DataFrame, partition_spec: PartitionSpec) -> DataFrame:

        def _persist_and_count(df: RayDataFrame) -> int:
            self.persist(df)
            return df.count()
        rdf = self._to_ray_df(df)
        num_funcs = {KEYWORD_ROWCOUNT: lambda: _persist_and_count(rdf), KEYWORD_PARALLELISM: lambda: self.get_current_parallelism()}
        num = partition_spec.get_num_partitions(**num_funcs)
        pdf = rdf.native
        if num > 0:
            if partition_spec.algo in ['default', 'hash', 'even', 'coarse']:
                pdf = pdf.repartition(num)
            elif partition_spec.algo == 'rand':
                pdf = pdf.repartition(num, shuffle=True)
            else:
                raise NotImplementedError(partition_spec.algo + ' is not supported')
        return RayDataFrame(pdf, schema=rdf.schema, internal_schema=True)

    def broadcast(self, df: DataFrame) -> DataFrame:
        return df

    def persist(self, df: DataFrame, lazy: bool=False, **kwargs: Any) -> DataFrame:
        df = self._to_auto_df(df)
        if isinstance(df, RayDataFrame):
            return df.persist(**kwargs)
        return df

    def convert_yield_dataframe(self, df: DataFrame, as_local: bool) -> DataFrame:
        if isinstance(df, RayDataFrame):
            return df if not as_local else df.as_local()
        return super().convert_yield_dataframe(df, as_local)

    def union(self, df1: DataFrame, df2: DataFrame, distinct: bool=True) -> DataFrame:
        if distinct:
            return super().union(df1, df2, distinct)
        assert_or_throw(df1.schema == df2.schema, ValueError(f'{df1.schema} != {df2.schema}'))
        tdf1 = self._to_ray_df(df1)
        tdf2 = self._to_ray_df(df2)
        return RayDataFrame(tdf1.native.union(tdf2.native), df1.schema)

    def load_df(self, path: Union[str, List[str]], format_hint: Any=None, columns: Any=None, **kwargs: Any) -> DataFrame:
        return self._io.load_df(uri=path, format_hint=format_hint, columns=columns, **kwargs)

    def save_df(self, df: DataFrame, path: str, format_hint: Any=None, mode: str='overwrite', partition_spec: Optional[PartitionSpec]=None, force_single: bool=False, **kwargs: Any) -> None:
        partition_spec = partition_spec or PartitionSpec()
        df = self._to_ray_df(df)
        self._io.save_df(df, uri=path, format_hint=format_hint, mode=mode, partition_spec=partition_spec, force_single=force_single, **kwargs)

    def _to_ray_df(self, df: Any, schema: Any=None) -> RayDataFrame:
        res = self._to_auto_df(df, schema)
        if not isinstance(res, RayDataFrame):
            return RayDataFrame(res)
        return res

    def _to_auto_df(self, df: Any, schema: Any=None) -> DataFrame:
        if isinstance(df, (DuckDataFrame, RayDataFrame)):
            assert_or_throw(schema is None, ValueError('schema must be None when df is a DataFrame'))
            return df
        return RayDataFrame(df, schema)

    def _get_remote_args(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        for k, v in self.conf.items():
            if k.startswith('fugue.ray.remote.'):
                key = k.split('.', 3)[-1]
                res[key] = v
        return res