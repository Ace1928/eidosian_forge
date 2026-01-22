import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
def _get_expr_stmt_definition_exprs(expr_stmt):
    exprs = []
    for list_ in expr_stmt.children[:-2:2]:
        if list_.type in ('testlist_star_expr', 'testlist'):
            exprs += _iter_definition_exprs_from_lists(list_)
        else:
            exprs.append(list_)
    return exprs