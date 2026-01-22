import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
def _is_bytes_literal(self, string):
    if string.type == 'fstring':
        return False
    return 'b' in string.string_prefix.lower()