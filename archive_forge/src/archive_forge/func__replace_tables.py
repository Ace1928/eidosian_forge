from __future__ import annotations
import datetime
import math
import numbers
import re
import textwrap
import typing as t
from collections import deque
from copy import deepcopy
from enum import auto
from functools import reduce
from sqlglot.errors import ErrorLevel, ParseError
from sqlglot.helper import (
from sqlglot.tokens import Token
def _replace_tables(node: Expression) -> Expression:
    if isinstance(node, Table):
        original = normalize_table_name(node, dialect=dialect)
        new_name = mapping.get(original)
        if new_name:
            table = to_table(new_name, **{k: v for k, v in node.args.items() if k not in TABLE_PARTS}, dialect=dialect)
            table.add_comments([original])
            return table
    return node