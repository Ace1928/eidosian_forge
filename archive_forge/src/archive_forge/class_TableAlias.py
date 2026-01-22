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
class TableAlias(Expression):
    arg_types = {'this': False, 'columns': False}

    @property
    def columns(self):
        return self.args.get('columns') or []