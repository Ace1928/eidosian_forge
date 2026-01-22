from __future__ import annotations
import typing as t
import uuid
from collections import defaultdict
import sqlglot
from sqlglot import Dialect, expressions as exp
from sqlglot.dataframe.sql import functions as F
from sqlglot.dataframe.sql.dataframe import DataFrame
from sqlglot.dataframe.sql.readwriter import DataFrameReader
from sqlglot.dataframe.sql.types import StructType
from sqlglot.dataframe.sql.util import get_column_mapping_from_schema_input
from sqlglot.helper import classproperty
from sqlglot.optimizer import optimize
from sqlglot.optimizer.qualify_columns import quote_identifiers
@property
def _auto_incrementing_name(self) -> str:
    name = f'a{self.incrementing_id}'
    self.incrementing_id += 1
    return name