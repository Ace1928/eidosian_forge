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
def _pl_special_verb(self, word: str, count: Optional[Union[str, int]]=None) -> Union[str, bool]:
    if self.classical_dict['zero'] and str(count).lower() in pl_count_zero:
        return False
    count = self.get_count(count)
    if count == 1:
        return word
    value = self.ud_match(word, self.pl_v_user_defined)
    if value is not None:
        return value
    try:
        words = Words(word)
    except IndexError:
        return False
    if words.first in plverb_irregular_pres:
        return f'{plverb_irregular_pres[words.first]}{words[len(words.first):]}'
    if words.first in plverb_irregular_non_pres:
        return word
    if words.first.endswith("n't") and words.first[:-3] in plverb_irregular_pres:
        return f"{plverb_irregular_pres[words.first[:-3]]}n't{words[len(words.first):]}"
    if words.first.endswith("n't"):
        return word
    mo = PLVERB_SPECIAL_S_RE.search(word)
    if mo:
        return False
    if WHITESPACE.search(word):
        return False
    if words.lowered == 'quizzes':
        return 'quiz'
    if words.lowered[-4:] in ('ches', 'shes', 'zzes', 'sses') or words.lowered[-3:] == 'xes':
        return words[:-2]
    if words.lowered[-3:] == 'ies' and len(words) > 3:
        return words.lowered[:-3] + 'y'
    if words.last.lower() in pl_v_oes_oe or words.lowered[-4:] in pl_v_oes_oe_endings_size4 or words.lowered[-5:] in pl_v_oes_oe_endings_size5:
        return words[:-1]
    if words.lowered.endswith('oes') and len(words) > 3:
        return words.lowered[:-2]
    mo = ENDS_WITH_S.search(words)
    if mo:
        return mo.group(1)
    return False