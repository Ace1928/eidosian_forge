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
class ParseJSON(Func):
    _sql_names = ['PARSE_JSON', 'JSON_PARSE']
    arg_types = {'this': True, 'expressions': False}
    is_var_len_args = True