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
class MagicMethodCompletion(BaseCompletionType):

    def matches(self, cursor_offset: int, line: str, *, current_block: Optional[str]=None, complete_magic_methods: Optional[bool]=None, **kwargs: Any) -> Optional[Set[str]]:
        if current_block is None or complete_magic_methods is None or (not complete_magic_methods):
            return None
        r = self.locate(cursor_offset, line)
        if r is None:
            return None
        if 'class' not in current_block:
            return None
        return {name for name in MAGIC_METHODS if name.startswith(r.word)}

    def locate(self, cursor_offset: int, line: str) -> Optional[LinePart]:
        return lineparts.current_method_definition_name(cursor_offset, line)