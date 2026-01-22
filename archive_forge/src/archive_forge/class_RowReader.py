from __future__ import annotations
import typing as t
from sqlglot.dialects.dialect import DialectType
from sqlglot.helper import dict_depth
from sqlglot.schema import AbstractMappingSchema, normalize_name
class RowReader:

    def __init__(self, columns, column_range=None):
        self.columns = {column: i for i, column in enumerate(columns) if not column_range or i in column_range}
        self.row = None

    def __getitem__(self, column):
        return self.row[self.columns[column]]