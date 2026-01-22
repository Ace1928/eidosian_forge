import __main__
import abc
import glob
import itertools
import keyword
import logging
import os
import re
import rlcompleter
import builtins
from enum import Enum
from typing import (
from . import inspection
from . import line as lineparts
from .line import LinePart
from .lazyre import LazyReCompile
from .simpleeval import safe_eval, evaluate_current_expression, EvaluationError
from .importcompletion import ModuleGatherer
class DictKeyCompletion(BaseCompletionType):

    def matches(self, cursor_offset: int, line: str, *, locals_: Optional[Dict[str, Any]]=None, **kwargs: Any) -> Optional[Set[str]]:
        if locals_ is None:
            return None
        r = self.locate(cursor_offset, line)
        if r is None:
            return None
        current_dict_parts = lineparts.current_dict(cursor_offset, line)
        if current_dict_parts is None:
            return None
        dexpr = current_dict_parts.word
        try:
            obj = safe_eval(dexpr, locals_)
        except EvaluationError:
            return None
        if isinstance(obj, dict) and obj.keys():
            matches = {f'{k!r}]' for k in obj.keys() if repr(k).startswith(r.word)}
            return matches if matches else None
        else:
            return None

    def locate(self, cursor_offset: int, line: str) -> Optional[LinePart]:
        return lineparts.current_dict_key(cursor_offset, line)

    def format(self, match: str) -> str:
        return match[:-1]