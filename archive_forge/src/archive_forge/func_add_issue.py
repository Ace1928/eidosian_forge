import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
def add_issue(self, node, code, message):
    line = node.start_pos[0]
    args = (code, message, node)
    self._error_dict.setdefault(line, args)