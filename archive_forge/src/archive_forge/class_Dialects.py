from __future__ import annotations
import logging
import typing as t
from enum import Enum, auto
from functools import reduce
from sqlglot import exp
from sqlglot.errors import ParseError
from sqlglot.generator import Generator
from sqlglot.helper import AutoName, flatten, is_int, seq_get
from sqlglot.jsonpath import parse as parse_json_path
from sqlglot.parser import Parser
from sqlglot.time import TIMEZONES, format_time
from sqlglot.tokens import Token, Tokenizer, TokenType
from sqlglot.trie import new_trie
class Dialects(str, Enum):
    """Dialects supported by SQLGLot."""
    DIALECT = ''
    ATHENA = 'athena'
    BIGQUERY = 'bigquery'
    CLICKHOUSE = 'clickhouse'
    DATABRICKS = 'databricks'
    DORIS = 'doris'
    DRILL = 'drill'
    DUCKDB = 'duckdb'
    HIVE = 'hive'
    MYSQL = 'mysql'
    ORACLE = 'oracle'
    POSTGRES = 'postgres'
    PRESTO = 'presto'
    PRQL = 'prql'
    REDSHIFT = 'redshift'
    SNOWFLAKE = 'snowflake'
    SPARK = 'spark'
    SPARK2 = 'spark2'
    SQLITE = 'sqlite'
    STARROCKS = 'starrocks'
    TABLEAU = 'tableau'
    TERADATA = 'teradata'
    TRINO = 'trino'
    TSQL = 'tsql'