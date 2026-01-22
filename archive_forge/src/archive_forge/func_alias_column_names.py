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
@property
def alias_column_names(self) -> t.List[str]:
    table_alias = self.args.get('alias')
    if not table_alias:
        return []
    return [c.name for c in table_alias.args.get('columns') or []]