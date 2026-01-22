import logging
from typing import Any, Dict, Iterable, List, Optional, Union
import duckdb
from duckdb import DuckDBPyConnection, DuckDBPyRelation
from triad import SerializableRLock
from triad.utils.assertion import assert_or_throw
from triad.utils.schema import quote_name
from fugue import (
from fugue.collections.partition import PartitionSpec, parse_presort_exp
from fugue.collections.sql import StructuredRawSQL, TempTableName
from fugue.dataframe import DataFrame, DataFrames, LocalBoundedDataFrame
from fugue.dataframe.utils import get_join_schemas
from ._io import DuckDBIO
from ._utils import (
from .dataframe import DuckDataFrame, _duck_as_arrow
class DuckExecutionEngine(ExecutionEngine):
    """The execution engine using DuckDB.
    Please read |ExecutionEngineTutorial| to understand this important Fugue concept

    :param conf: |ParamsLikeObject|, read |FugueConfig| to learn Fugue specific options
    :param connection: DuckDB connection
    """

    def __init__(self, conf: Any=None, connection: Optional[DuckDBPyConnection]=None):
        super().__init__(conf)
        self._native_engine = NativeExecutionEngine(conf)
        self._con = connection or duckdb.connect()
        self._external_con = connection is not None
        self._context_lock = SerializableRLock()
        self._registered_dfs: Dict[str, DuckDataFrame] = {}
        try:
            for pg in list(self._get_pragmas()):
                self._con.execute(pg)
            for ext in self.conf.get(_FUGUE_DUCKDB_EXTENSIONS, '').split(','):
                _ext = ext.strip()
                if _ext != '':
                    self._con.install_extension(_ext)
                    self._con.load_extension(_ext)
        except Exception:
            self.stop()
            raise

    def _get_pragmas(self) -> Iterable[str]:
        for k, v in self.conf.items():
            if k.startswith(_FUGUE_DUCKDB_PRAGMA_CONFIG_PREFIX):
                name = k[len(_FUGUE_DUCKDB_PRAGMA_CONFIG_PREFIX):]
                assert_or_throw(name.isidentifier(), ValueError(f'{name} is not a valid pragma key'))
                value = encode_value_to_expr(v)
                yield f'PRAGMA {name}={value};'

    def stop_engine(self) -> None:
        if not self._external_con:
            self._con.close()

    def __repr__(self) -> str:
        return 'DuckExecutionEngine'

    @property
    def is_distributed(self) -> bool:
        return False

    @property
    def connection(self) -> DuckDBPyConnection:
        return self._con

    @property
    def log(self) -> logging.Logger:
        return self._native_engine.log

    def create_default_sql_engine(self) -> SQLEngine:
        return DuckDBEngine(self)

    def create_default_map_engine(self) -> MapEngine:
        return PandasMapEngine(self._native_engine)

    def get_current_parallelism(self) -> int:
        return 1

    def to_df(self, df: Any, schema: Any=None) -> DataFrame:
        return _to_duck_df(self, df, schema=schema)

    def repartition(self, df: DataFrame, partition_spec: PartitionSpec) -> DataFrame:
        self.log.warning("%s doesn't respect repartition", self)
        return df

    def broadcast(self, df: DataFrame) -> DataFrame:
        return df

    def persist(self, df: DataFrame, lazy: bool=False, **kwargs: Any) -> DataFrame:
        if isinstance(df, DuckDataFrame):
            res: DataFrame = ArrowDataFrame(df.as_arrow())
        else:
            res = self.to_df(df)
        res.reset_metadata(df.metadata)
        return res

    def join(self, df1: DataFrame, df2: DataFrame, how: str, on: Optional[List[str]]=None) -> DataFrame:
        key_schema, output_schema = get_join_schemas(df1, df2, how=how, on=on)
        t1, t2, t3 = (TempTableName(), TempTableName(), TempTableName())
        on_fields = ' AND '.join((f'{t1}.{encode_column_name(k)}={t2}.{encode_column_name(k)}' for k in key_schema))
        join_type = self._how_to_join(how)
        if how.lower() == 'cross':
            select_fields = ','.join((f'{t1}.{encode_column_name(k)}' if k in df1.schema else f'{t2}.{encode_column_name(k)}' for k in output_schema.names))
            sql = f'SELECT {select_fields} FROM {t1} {join_type} {t2}'
        elif how.lower() == 'right_outer':
            select_fields = ','.join((f'{t2}.{encode_column_name(k)}' if k in df2.schema else f'{t1}.{encode_column_name(k)}' for k in output_schema.names))
            sql = f'SELECT {select_fields} FROM {t2} LEFT OUTER JOIN {t1} ON {on_fields}'
        elif how.lower() == 'full_outer':
            select_fields = ','.join((f'COALESCE({t1}.{encode_column_name(k)},{t2}.{encode_column_name(k)}) AS {encode_column_name(k)}' if k in key_schema else encode_column_name(k) for k in output_schema.names))
            sql = f'SELECT {select_fields} FROM {t1} {join_type} {t2} ON {on_fields}'
        elif how.lower() in ['semi', 'left_semi']:
            keys = ','.join(encode_schema_names(key_schema))
            on_fields = ' AND '.join((f'{t1}.{encode_column_name(k)}={t3}.{encode_column_name(k)}' for k in key_schema))
            sql = f'SELECT {t1}.* FROM {t1} INNER JOIN (SELECT DISTINCT {keys} FROM {t2}) AS {t3} ON {on_fields}'
        elif how.lower() in ['anti', 'left_anti']:
            keys = ','.join(encode_schema_names(key_schema))
            on_fields = ' AND '.join((f'{t1}.{encode_column_name(k)}={t3}.{encode_column_name(k)}' for k in key_schema))
            sql = f'SELECT {t1}.* FROM {t1} LEFT OUTER JOIN (SELECT DISTINCT {keys}, 1 AS __contain__ FROM {t2}) AS {t3} ON {on_fields} WHERE {t3}.__contain__ IS NULL'
        else:
            select_fields = ','.join((f'{t1}.{encode_column_name(k)}' if k in df1.schema else f'{t2}.{encode_column_name(k)}' for k in output_schema.names))
            sql = f'SELECT {select_fields} FROM {t1} {join_type} {t2} ON {on_fields}'
        return self._sql(sql, {t1.key: df1, t2.key: df2})

    def _how_to_join(self, how: str):
        return how.upper().replace('_', ' ') + ' JOIN'

    def union(self, df1: DataFrame, df2: DataFrame, distinct: bool=True) -> DataFrame:
        assert_or_throw(df1.schema == df2.schema, ValueError(f'{df1.schema} != {df2.schema}'))
        if distinct:
            t1, t2 = (TempTableName(), TempTableName())
            sql = f'SELECT * FROM {t1} UNION SELECT * FROM {t2}'
            return self._sql(sql, {t1.key: df1, t2.key: df2})
        return DuckDataFrame(_to_duck_df(self, df1).native.union(_to_duck_df(self, df2).native))

    def subtract(self, df1: DataFrame, df2: DataFrame, distinct: bool=True) -> DataFrame:
        if distinct:
            t1, t2 = (TempTableName(), TempTableName())
            sql = f'SELECT * FROM {t1} EXCEPT SELECT * FROM {t2}'
            return self._sql(sql, {t1.key: df1, t2.key: df2})
        return DuckDataFrame(_to_duck_df(self, df1).native.except_(_to_duck_df(self, df2).native))

    def intersect(self, df1: DataFrame, df2: DataFrame, distinct: bool=True) -> DataFrame:
        if distinct:
            t1, t2 = (TempTableName(), TempTableName())
            sql = f'SELECT * FROM {t1} INTERSECT DISTINCT SELECT * FROM {t2}'
            return self._sql(sql, {t1.key: df1, t2.key: df2})
        raise NotImplementedError("DuckDB doesn't have consist behavior on INTERSECT ALL, so Fugue doesn't support it")

    def distinct(self, df: DataFrame) -> DataFrame:
        rel = _to_duck_df(self, df).native.distinct()
        return DuckDataFrame(rel)

    def dropna(self, df: DataFrame, how: str='any', thresh: int=None, subset: List[str]=None) -> DataFrame:
        schema = df.schema
        if subset is not None:
            schema = schema.extract(subset)
        if how == 'all':
            thr = 0
        elif how == 'any':
            thr = thresh or len(schema)
        else:
            raise ValueError(f'{how} is not one of any and all')
        cw = [f'CASE WHEN {encode_column_name(f)} IS NULL THEN 0 ELSE 1 END' for f in schema.names]
        expr = ' + '.join(cw) + f' >= {thr}'
        return DuckDataFrame(_to_duck_df(self, df).native.filter(expr))

    def fillna(self, df: DataFrame, value: Any, subset: List[str]=None) -> DataFrame:

        def _build_value_dict(names: List[str]) -> Dict[str, str]:
            if not isinstance(value, dict):
                v = encode_value_to_expr(value)
                return {n: v for n in names}
            else:
                return {n: encode_value_to_expr(value[n]) for n in names}
        names = df.columns
        if isinstance(value, dict):
            names = list(value.keys())
        elif subset is not None:
            names = list(df.schema.extract(subset).names)
        vd = _build_value_dict(names)
        assert_or_throw(all((v != 'NULL' for v in vd.values())), ValueError('fillna value can not be None or contain None'))
        cols = [f'COALESCE({encode_column_name(f)}, {vd[f]}) AS {encode_column_name(f)}' if f in names else encode_column_name(f) for f in df.columns]
        return DuckDataFrame(_to_duck_df(self, df).native.project(', '.join(cols)))

    def sample(self, df: DataFrame, n: Optional[int]=None, frac: Optional[float]=None, replace: bool=False, seed: Optional[int]=None) -> DataFrame:
        assert_or_throw(n is None and frac is not None and (frac >= 0.0) or (frac is None and n is not None and (n >= 0)), ValueError(f'one and only one of n and frac should be non-negative, {n}, {frac}'))
        tb = TempTableName()
        if frac is not None:
            sql = f'SELECT * FROM {tb} USING SAMPLE bernoulli({frac * 100} PERCENT)'
        else:
            sql = f'SELECT * FROM {tb} USING SAMPLE reservoir({n} ROWS)'
        if seed is not None:
            sql += f' REPEATABLE ({seed})'
        return self._sql(sql, {tb.key: df})

    def take(self, df: DataFrame, n: int, presort: str, na_position: str='last', partition_spec: Optional[PartitionSpec]=None) -> DataFrame:
        partition_spec = partition_spec or PartitionSpec()
        assert_or_throw(isinstance(n, int), ValueError('n needs to be an integer'))
        if presort is not None and presort != '':
            _presort = parse_presort_exp(presort)
        else:
            _presort = partition_spec.presort
        tb = TempTableName()
        if len(_presort) == 0:
            if len(partition_spec.partition_by) == 0:
                return DuckDataFrame(_to_duck_df(self, df).native.limit(n))
            cols = ', '.join(encode_schema_names(df.schema))
            pcols = ', '.join(encode_column_names(partition_spec.partition_by))
            sql = f'SELECT *, ROW_NUMBER() OVER (PARTITION BY {pcols}) AS __fugue_take_param FROM {tb}'
            sql = f'SELECT {cols} FROM ({sql}) WHERE __fugue_take_param<={n}'
            return self._sql(sql, {tb.key: df})
        sorts: List[str] = []
        for k, v in _presort.items():
            s = encode_column_name(k)
            if not v:
                s += ' DESC'
            s += ' NULLS FIRST' if na_position == 'first' else ' NULLS LAST'
            sorts.append(s)
        sort_expr = 'ORDER BY ' + ', '.join(sorts)
        if len(partition_spec.partition_by) == 0:
            sql = f'SELECT * FROM {tb} {sort_expr} LIMIT {n}'
            return self._sql(sql, {tb.key: df})
        cols = ', '.join(encode_schema_names(df.schema))
        pcols = ', '.join(encode_column_names(partition_spec.partition_by))
        sql = f'SELECT *, ROW_NUMBER() OVER (PARTITION BY {pcols} {sort_expr}) AS __fugue_take_param FROM {tb}'
        sql = f'SELECT {cols} FROM ({sql}) WHERE __fugue_take_param<={n}'
        return self._sql(sql, {tb.key: df})

    def load_df(self, path: Union[str, List[str]], format_hint: Any=None, columns: Any=None, **kwargs: Any) -> LocalBoundedDataFrame:
        dio = DuckDBIO(self.connection)
        return dio.load_df(path, format_hint, columns, **kwargs)

    def save_df(self, df: DataFrame, path: str, format_hint: Any=None, mode: str='overwrite', partition_spec: Optional[PartitionSpec]=None, force_single: bool=False, **kwargs: Any) -> None:
        partition_spec = partition_spec or PartitionSpec()
        if not partition_spec.empty and (not force_single):
            kwargs['partition_cols'] = partition_spec.partition_by
        dio = DuckDBIO(self.connection)
        dio.save_df(_to_duck_df(self, df), path, format_hint, mode, **kwargs)

    def convert_yield_dataframe(self, df: DataFrame, as_local: bool) -> DataFrame:
        if as_local:
            return df.as_local()
        return df.as_local() if self._ctx_count <= 1 and (not self._external_con) else df

    def _sql(self, sql: str, dfs: Dict[str, DataFrame]) -> DuckDataFrame:
        with self._context_lock:
            df = self.sql_engine.select(DataFrames(dfs), StructuredRawSQL.from_expr(sql, dialect=None))
            return DuckDataFrame(df.native)