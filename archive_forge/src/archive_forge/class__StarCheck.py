import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
@ErrorFinder.register_rule(value='*')
class _StarCheck(SyntaxRule):
    message = 'named arguments must follow bare *'

    def is_issue(self, leaf):
        params = leaf.parent
        if params.type == 'parameters' and params:
            after = params.children[params.children.index(leaf) + 1:]
            after = [child for child in after if child not in (',', ')') and (not child.star_count)]
            return len(after) == 0