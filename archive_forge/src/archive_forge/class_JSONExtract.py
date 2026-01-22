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
class JSONExtract(Binary, Func):
    arg_types = {'this': True, 'expression': True, 'only_json_types': False, 'expressions': False}
    _sql_names = ['JSON_EXTRACT']
    is_var_len_args = True

    @property
    def output_name(self) -> str:
        return self.expression.output_name if not self.expressions else ''