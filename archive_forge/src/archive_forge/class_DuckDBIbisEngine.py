from typing import Any, Callable, Dict, Optional, Tuple
import ibis
from ibis.backends.pandas import Backend
from fugue import DataFrame, DataFrames, ExecutionEngine
from fugue.collections.sql import StructuredRawSQL, TempTableName
from fugue_ibis import IbisTable
from fugue_ibis._utils import to_ibis_schema
from fugue_ibis.execution.ibis_engine import IbisEngine, parse_ibis_engine
from .execution_engine import DuckDBEngine, DuckExecutionEngine
class DuckDBIbisEngine(IbisEngine):

    def select(self, dfs: DataFrames, ibis_func: Callable[[ibis.BaseBackend], IbisTable]) -> DataFrame:
        be = _BackendWrapper().connect({})
        be.set_schemas(dfs)
        expr = ibis_func(be)
        sql = StructuredRawSQL.from_expr(str(ibis.postgres.compile(expr).compile(compile_kwargs={'literal_binds': True})), prefix='"<tmpdf:', suffix='>"', dialect='postgres')
        engine = DuckDBEngine(self.execution_engine)
        _dfs = DataFrames({be._name_map[k][0].key: v for k, v in dfs.items()})
        return engine.select(_dfs, sql)