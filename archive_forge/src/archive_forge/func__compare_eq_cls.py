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
def _compare_eq_cls(left: Any, right: Any, highlighter: _HighlightFunc, verbose: int) -> List[str]:
    if not has_default_eq(left):
        return []
    if isdatacls(left):
        import dataclasses
        all_fields = dataclasses.fields(left)
        fields_to_check = [info.name for info in all_fields if info.compare]
    elif isattrs(left):
        all_fields = left.__attrs_attrs__
        fields_to_check = [field.name for field in all_fields if getattr(field, 'eq')]
    elif isnamedtuple(left):
        fields_to_check = left._fields
    else:
        assert False
    indent = '  '
    same = []
    diff = []
    for field in fields_to_check:
        if getattr(left, field) == getattr(right, field):
            same.append(field)
        else:
            diff.append(field)
    explanation = []
    if same or diff:
        explanation += ['']
    if same and verbose < 2:
        explanation.append('Omitting %s identical items, use -vv to show' % len(same))
    elif same:
        explanation += ['Matching attributes:']
        explanation += highlighter(pprint.pformat(same)).splitlines()
    if diff:
        explanation += ['Differing attributes:']
        explanation += highlighter(pprint.pformat(diff)).splitlines()
        for field in diff:
            field_left = getattr(left, field)
            field_right = getattr(right, field)
            explanation += ['', f'Drill down into differing attribute {field}:', f'{indent}{field}: {highlighter(repr(field_left))} != {highlighter(repr(field_right))}']
            explanation += [indent + line for line in _compare_eq_any(field_left, field_right, highlighter, verbose)]
    return explanation