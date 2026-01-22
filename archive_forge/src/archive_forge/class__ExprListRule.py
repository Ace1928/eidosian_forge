import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
@ErrorFinder.register_rule(type='expr_list')
class _ExprListRule(_CheckAssignmentRule):

    def is_issue(self, expr_list):
        for expr in expr_list.children[::2]:
            self._check_assignment(expr)