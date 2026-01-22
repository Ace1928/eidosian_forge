import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
def add_context(self, node):
    return _Context(node, self._add_syntax_error, parent_context=self)