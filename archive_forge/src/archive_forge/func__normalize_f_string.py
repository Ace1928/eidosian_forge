import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import (
from mypy_extensions import trait
from black.comments import contains_pragma_comment
from black.lines import Line, append_leaves
from black.mode import Feature, Mode, Preview
from black.nodes import (
from black.rusty import Err, Ok, Result
from black.strings import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def _normalize_f_string(self, string: str, prefix: str) -> str:
    """
        Pre-Conditions:
            * assert_is_leaf_string(@string)

        Returns:
            * If @string is an f-string that contains no f-expressions, we
            return a string identical to @string except that the 'f' prefix
            has been stripped and all double braces (i.e. '{{' or '}}') have
            been normalized (i.e. turned into '{' or '}').
                OR
            * Otherwise, we return @string.
        """
    assert_is_leaf_string(string)
    if 'f' in prefix and (not fstring_contains_expr(string)):
        new_prefix = prefix.replace('f', '')
        temp = string[len(prefix):]
        temp = re.sub('\\{\\{', '{', temp)
        temp = re.sub('\\}\\}', '}', temp)
        new_string = temp
        return f'{new_prefix}{new_string}'
    else:
        return string