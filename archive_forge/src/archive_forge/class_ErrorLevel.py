from __future__ import annotations
import typing as t
from enum import auto
from sqlglot.helper import AutoName
class ErrorLevel(AutoName):
    IGNORE = auto()
    'Ignore all errors.'
    WARN = auto()
    'Log all errors.'
    RAISE = auto()
    'Collect all errors and raise a single exception.'
    IMMEDIATE = auto()
    'Immediately raise an exception on the first error found.'