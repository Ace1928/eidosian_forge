import re
import sys
from collections import namedtuple
from typing import List, Dict, Any
from lark.tree import Meta
from lark.visitors import Transformer, Discard, _DiscardType, v_args
def to_string_dollar(self, value: Any) -> Any:
    """Wrap a string in ${ and }"""
    if isinstance(value, str):
        if value.startswith('"') and value.endswith('"'):
            return str(value)[1:-1]
        return f'${{{value}}}'
    return value