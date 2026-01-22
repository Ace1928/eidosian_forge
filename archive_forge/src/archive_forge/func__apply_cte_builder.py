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
def _apply_cte_builder(instance: E, alias: ExpOrStr, as_: ExpOrStr, recursive: t.Optional[bool]=None, append: bool=True, dialect: DialectType=None, copy: bool=True, **opts) -> E:
    alias_expression = maybe_parse(alias, dialect=dialect, into=TableAlias, **opts)
    as_expression = maybe_parse(as_, dialect=dialect, **opts)
    cte = CTE(this=as_expression, alias=alias_expression)
    return _apply_child_list_builder(cte, instance=instance, arg='with', append=append, copy=copy, into=With, properties={'recursive': recursive or False})