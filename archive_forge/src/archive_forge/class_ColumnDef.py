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
class ColumnDef(Expression):
    arg_types = {'this': True, 'kind': False, 'constraints': False, 'exists': False, 'position': False}

    @property
    def constraints(self) -> t.List[ColumnConstraint]:
        return self.args.get('constraints') or []

    @property
    def kind(self) -> t.Optional[DataType]:
        return self.args.get('kind')