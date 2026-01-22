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
class QueryTransform(Expression):
    arg_types = {'expressions': True, 'command_script': True, 'schema': False, 'row_format_before': False, 'record_writer': False, 'row_format_after': False, 'record_reader': False}