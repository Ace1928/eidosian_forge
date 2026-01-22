import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
def _skip_parens_bottom_up(node):
    """
    Returns an ancestor node of an expression, skipping all levels of parens
    bottom-up.
    """
    while node.parent is not None:
        node = node.parent
        if node.type != 'atom' or node.children[0] != '(':
            return node
    return None