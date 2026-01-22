from __future__ import annotations
import abc
import datetime
import contextlib
from pydantic import BaseModel, model_validator, Field, PrivateAttr, validator
from lazyops.utils.logs import logger
from .static import SqliteTemplates
from .registry import get_or_register_sqlite_schema, get_or_register_sqlite_connection, retrieve_sqlite_model_schema, get_sqlite_model_pkey, get_or_register_sqlite_tablename, SQLiteModelConfig, get_sqlite_model_config
from .utils import normalize_sql_text
from typing import Optional, List, Tuple, Dict, Union, TypeVar, Any, overload, TYPE_CHECKING
@classmethod
def _create_sql_query_from_kwargs(cls, schemas: Dict[str, Union[str, List[str], Dict[str, str]]], query: Optional[str]=None, **kwargs) -> str:
    """
        Creates the sql query from the kwargs

        precision: The search mode (0: loose, 1: field, 2: precise)
        """
    loose_matches: List[str] = []
    field_matches: List[str] = []
    exact_matches: List[str] = []
    search_precisions = schemas['search_precisions']
    for k, v in kwargs.items():
        if v is None:
            continue
        if isinstance(v, str) and (not v.strip()):
            continue
        if k not in schemas['sql_fields']:
            continue
        precision = search_precisions.get(k, k)
        if precision == 0:
            loose_matches.append(normalize_sql_text(v))
        elif precision == 1:
            field_matches.append(f'({k} : {normalize_sql_text(v)})')
        elif precision == 2:
            exact_matches.append(f'{k} = "{v}"')
    query = query or ''
    tablename = schemas['tablename']
    if not loose_matches and (not field_matches) and (not exact_matches) and query:
        query = f'{tablename}_fts MATCH "{query}"'
        return query
    if exact_matches:
        query += f'({' OR '.join(exact_matches)}) AND '
    if loose_matches or field_matches:
        query += f'{tablename}_fts MATCH "'
        if field_matches:
            query += f'{' OR '.join(field_matches)} AND '
        if loose_matches:
            query += f'{' OR '.join(loose_matches)}'
        query = query.rstrip(' AND ')
        query += '"'
    query = query.replace('AND OR ', 'AND ')
    query = query.rstrip(' AND ')
    return query