import ast
import collections
import contextlib
import functools
import itertools
import re
from numbers import Number
from typing import (
from more_itertools import windowed_complete
from typeguard import typechecked
from typing_extensions import Annotated, Literal
def _plequal(self, word1: str, word2: str, pl) -> Union[str, bool]:
    classval = self.classical_dict.copy()
    self.classical_dict = all_classical.copy()
    if word1 == word2:
        return 'eq'
    if word1 == pl(word2):
        return 'p:s'
    if pl(word1) == word2:
        return 's:p'
    self.classical_dict = no_classical.copy()
    if word1 == pl(word2):
        return 'p:s'
    if pl(word1) == word2:
        return 's:p'
    self.classical_dict = classval.copy()
    if pl == self.plural or pl == self.plural_noun:
        if self._pl_check_plurals_N(word1, word2):
            return 'p:p'
        if self._pl_check_plurals_N(word2, word1):
            return 'p:p'
    if pl == self.plural or pl == self.plural_adj:
        if self._pl_check_plurals_adj(word1, word2):
            return 'p:p'
    return False