import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
@ErrorFinder.register_rule(type='error_node')
class _InvalidSyntaxRule(SyntaxRule):
    message = 'invalid syntax'
    fstring_message = 'f-string: invalid syntax'

    def get_node(self, node):
        return node.get_next_leaf()

    def is_issue(self, node):
        error = node.get_next_leaf().type != 'error_leaf'
        if error and _any_fstring_error(self._normalizer.version, node):
            self.add_issue(node, message=self.fstring_message)
        else:
            return error