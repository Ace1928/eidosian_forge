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
def _apply_conjunction_builder(*expressions, instance, arg, into=None, append=True, copy=True, dialect=None, **opts):
    expressions = [exp for exp in expressions if exp is not None and exp != '']
    if not expressions:
        return instance
    inst = maybe_copy(instance, copy)
    existing = inst.args.get(arg)
    if append and existing is not None:
        expressions = [existing.this if into else existing] + list(expressions)
    node = and_(*expressions, dialect=dialect, copy=copy, **opts)
    inst.set(arg, into(this=node) if into else node)
    return inst