import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
def _get_for_stmt_definition_exprs(for_stmt):
    exprlist = for_stmt.children[1]
    return list(_iter_definition_exprs_from_lists(exprlist))