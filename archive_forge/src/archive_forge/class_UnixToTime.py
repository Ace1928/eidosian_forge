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
class UnixToTime(Func):
    arg_types = {'this': True, 'scale': False, 'zone': False, 'hours': False, 'minutes': False, 'format': False}
    SECONDS = Literal.number(0)
    DECIS = Literal.number(1)
    CENTIS = Literal.number(2)
    MILLIS = Literal.number(3)
    DECIMILLIS = Literal.number(4)
    CENTIMILLIS = Literal.number(5)
    MICROS = Literal.number(6)
    DECIMICROS = Literal.number(7)
    CENTIMICROS = Literal.number(8)
    NANOS = Literal.number(9)