import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
@ErrorFinder.register_rule(value='**')
class _StarStarCheck(SyntaxRule):
    message = 'dict unpacking cannot be used in dict comprehension'

    def is_issue(self, leaf):
        if leaf.parent.type == 'dictorsetmaker':
            comp_for = leaf.get_next_sibling().get_next_sibling()
            return comp_for is not None and comp_for.type in _COMP_FOR_TYPES