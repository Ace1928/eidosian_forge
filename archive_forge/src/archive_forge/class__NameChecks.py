import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
@ErrorFinder.register_rule(type='name')
class _NameChecks(SyntaxRule):
    message = 'cannot assign to __debug__'
    message_none = 'cannot assign to None'

    def is_issue(self, leaf):
        self._normalizer.context.add_name(leaf)
        if leaf.value == '__debug__' and leaf.is_definition():
            return True