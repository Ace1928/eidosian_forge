import railroad
from pip._vendor import pyparsing
import typing
from typing import (
from jinja2 import Template
from io import StringIO
import inspect
def _should_vertical(specification: int, exprs: Iterable[pyparsing.ParserElement]) -> bool:
    """
    Returns true if we should return a vertical list of elements
    """
    if specification is None:
        return False
    else:
        return len(_visible_exprs(exprs)) >= specification