from __future__ import annotations
import re
from typing import Any
from typing import Optional
from typing import TypeVar
from .operators import CONTAINED_BY
from .operators import CONTAINS
from .operators import OVERLAP
from ... import types as sqltypes
from ... import util
from ...sql import expression
from ...sql import operators
from ...sql._typing import _TypeEngineArgument
def _split_enum_values(array_string):
    if '"' not in array_string:
        return array_string.split(',') if array_string else []
    text = array_string.replace('\\"', '_$ESC_QUOTE$_')
    text = text.replace('\\\\', '\\')
    result = []
    on_quotes = re.split('(")', text)
    in_quotes = False
    for tok in on_quotes:
        if tok == '"':
            in_quotes = not in_quotes
        elif in_quotes:
            result.append(tok.replace('_$ESC_QUOTE$_', '"'))
        else:
            result.extend(re.findall('([^\\s,]+),?', tok))
    return result