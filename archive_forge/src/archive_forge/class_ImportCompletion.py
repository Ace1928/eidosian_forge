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
class ImportCompletion(BaseCompletionType):

    def __init__(self, module_gatherer: ModuleGatherer, mode: AutocompleteModes=AutocompleteModes.SIMPLE):
        super().__init__(False, mode)
        self.module_gatherer = module_gatherer

    def matches(self, cursor_offset: int, line: str, **kwargs: Any) -> Optional[Set[str]]:
        return self.module_gatherer.complete(cursor_offset, line)

    def locate(self, cursor_offset: int, line: str) -> Optional[LinePart]:
        return lineparts.current_word(cursor_offset, line)

    def format(self, word: str) -> str:
        return _after_last_dot(word)