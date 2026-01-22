import railroad
from pip._vendor import pyparsing
import typing
from typing import (
from jinja2 import Template
from io import StringIO
import inspect
def _visible_exprs(exprs: Iterable[pyparsing.ParserElement]):
    non_diagramming_exprs = (pyparsing.ParseElementEnhance, pyparsing.PositionToken, pyparsing.And._ErrorStop)
    return [e for e in exprs if not (e.customName or e.resultsName or isinstance(e, non_diagramming_exprs))]