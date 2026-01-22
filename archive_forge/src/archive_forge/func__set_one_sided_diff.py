import collections.abc
import os
import pprint
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import Iterable
from typing import List
from typing import Literal
from typing import Mapping
from typing import Optional
from typing import Protocol
from typing import Sequence
from unicodedata import normalize
from _pytest import outcomes
import _pytest._code
from _pytest._io.pprint import PrettyPrinter
from _pytest._io.saferepr import saferepr
from _pytest._io.saferepr import saferepr_unlimited
from _pytest.config import Config
def _set_one_sided_diff(posn: str, set1: AbstractSet[Any], set2: AbstractSet[Any], highlighter: _HighlightFunc) -> List[str]:
    explanation = []
    diff = set1 - set2
    if diff:
        explanation.append(f'Extra items in the {posn} set:')
        for item in diff:
            explanation.append(highlighter(saferepr(item)))
    return explanation