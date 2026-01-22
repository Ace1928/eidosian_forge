import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
@ErrorFinder.register_rule(type='import_as_names')
class _TrailingImportComma(SyntaxRule):
    message = 'trailing comma not allowed without surrounding parentheses'

    def is_issue(self, node):
        if node.children[-1] == ',' and node.parent.children[-1] != ')':
            return True