import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
@ErrorFinder.register_rule(type='parameters')
@ErrorFinder.register_rule(type='lambdef')
class _ParameterRule(SyntaxRule):
    message = 'non-default argument follows default argument'

    def is_issue(self, node):
        param_names = set()
        default_only = False
        star_seen = False
        for p in _iter_params(node):
            if p.type == 'operator':
                if p.value == '*':
                    star_seen = True
                    default_only = False
                continue
            if p.name.value in param_names:
                message = "duplicate argument '%s' in function definition"
                self.add_issue(p.name, message=message % p.name.value)
            param_names.add(p.name.value)
            if not star_seen:
                if p.default is None and (not p.star_count):
                    if default_only:
                        return True
                elif p.star_count:
                    star_seen = True
                    default_only = False
                else:
                    default_only = True