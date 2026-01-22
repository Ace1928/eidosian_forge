from __future__ import annotations
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Union
import sqlalchemy
from langchain_core._api import deprecated
from langchain_core.utils import get_from_env
from sqlalchemy import (
from sqlalchemy.engine import Engine, Result
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
from sqlalchemy.schema import CreateTable
from sqlalchemy.sql.expression import Executable
from sqlalchemy.types import NullType
def _get_table_indexes(self, table: Table) -> str:
    indexes = self._inspector.get_indexes(table.name)
    indexes_formatted = '\n'.join(map(_format_index, indexes))
    return f'Table Indexes:\n{indexes_formatted}'