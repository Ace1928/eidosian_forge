import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
@ErrorFinder.register_rule(type='strings')
class _BytesAndStringMix(SyntaxRule):
    message = 'cannot mix bytes and nonbytes literals'

    def _is_bytes_literal(self, string):
        if string.type == 'fstring':
            return False
        return 'b' in string.string_prefix.lower()

    def is_issue(self, node):
        first = node.children[0]
        first_is_bytes = self._is_bytes_literal(first)
        for string in node.children[1:]:
            if first_is_bytes != self._is_bytes_literal(string):
                return True