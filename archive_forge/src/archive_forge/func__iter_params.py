import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
def _iter_params(parent_node):
    return (n for n in parent_node.children if n.type == 'param' or n.type == 'operator')