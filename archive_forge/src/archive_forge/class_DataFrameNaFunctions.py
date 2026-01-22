from __future__ import annotations
import functools
import logging
import typing as t
import zlib
from copy import copy
import sqlglot
from sqlglot import Dialect, expressions as exp
from sqlglot.dataframe.sql import functions as F
from sqlglot.dataframe.sql.column import Column
from sqlglot.dataframe.sql.group import GroupedData
from sqlglot.dataframe.sql.normalize import normalize
from sqlglot.dataframe.sql.operations import Operation, operation
from sqlglot.dataframe.sql.readwriter import DataFrameWriter
from sqlglot.dataframe.sql.transforms import replace_id_value
from sqlglot.dataframe.sql.util import get_tables_from_expression_with_join
from sqlglot.dataframe.sql.window import Window
from sqlglot.helper import ensure_list, object_to_dict, seq_get
class DataFrameNaFunctions:

    def __init__(self, df: DataFrame):
        self.df = df

    def drop(self, how: str='any', thresh: t.Optional[int]=None, subset: t.Optional[t.Union[str, t.Tuple[str, ...], t.List[str]]]=None) -> DataFrame:
        return self.df.dropna(how=how, thresh=thresh, subset=subset)

    def fill(self, value: t.Union[int, bool, float, str, t.Dict[str, t.Any]], subset: t.Optional[t.Union[str, t.Tuple[str, ...], t.List[str]]]=None) -> DataFrame:
        return self.df.fillna(value=value, subset=subset)

    def replace(self, to_replace: t.Union[bool, int, float, str, t.List, t.Dict], value: t.Optional[t.Union[bool, int, float, str, t.List]]=None, subset: t.Optional[t.Union[str, t.List[str]]]=None) -> DataFrame:
        return self.df.replace(to_replace=to_replace, value=value, subset=subset)