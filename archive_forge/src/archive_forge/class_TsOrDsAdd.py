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
class TsOrDsAdd(Func, TimeUnit):
    arg_types = {'this': True, 'expression': True, 'unit': False, 'return_type': False}

    @property
    def return_type(self) -> DataType:
        return DataType.build(self.args.get('return_type') or DataType.Type.DATE)