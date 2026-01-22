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
def _compare_eq_any(left: Any, right: Any, highlighter: _HighlightFunc, verbose: int=0) -> List[str]:
    explanation = []
    if istext(left) and istext(right):
        explanation = _diff_text(left, right, verbose)
    else:
        from _pytest.python_api import ApproxBase
        if isinstance(left, ApproxBase) or isinstance(right, ApproxBase):
            approx_side = left if isinstance(left, ApproxBase) else right
            other_side = right if isinstance(left, ApproxBase) else left
            explanation = approx_side._repr_compare(other_side)
        elif type(left) is type(right) and (isdatacls(left) or isattrs(left) or isnamedtuple(left)):
            explanation = _compare_eq_cls(left, right, highlighter, verbose)
        elif issequence(left) and issequence(right):
            explanation = _compare_eq_sequence(left, right, highlighter, verbose)
        elif isset(left) and isset(right):
            explanation = _compare_eq_set(left, right, highlighter, verbose)
        elif isdict(left) and isdict(right):
            explanation = _compare_eq_dict(left, right, highlighter, verbose)
        if isiterable(left) and isiterable(right):
            expl = _compare_eq_iterable(left, right, highlighter, verbose)
            explanation.extend(expl)
    return explanation