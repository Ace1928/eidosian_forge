import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
@ErrorFinder.register_rule(type='fstring')
class _FStringRule(SyntaxRule):
    _fstring_grammar = None
    message_expr = 'f-string expression part cannot include a backslash'
    message_nested = 'f-string: expressions nested too deeply'
    message_conversion = "f-string: invalid conversion character: expected 's', 'r', or 'a'"

    def _check_format_spec(self, format_spec, depth):
        self._check_fstring_contents(format_spec.children[1:], depth)

    def _check_fstring_expr(self, fstring_expr, depth):
        if depth >= 2:
            self.add_issue(fstring_expr, message=self.message_nested)
        expr = fstring_expr.children[1]
        if '\\' in expr.get_code():
            self.add_issue(expr, message=self.message_expr)
        children_2 = fstring_expr.children[2]
        if children_2.type == 'operator' and children_2.value == '=':
            conversion = fstring_expr.children[3]
        else:
            conversion = children_2
        if conversion.type == 'fstring_conversion':
            name = conversion.children[1]
            if name.value not in ('s', 'r', 'a'):
                self.add_issue(name, message=self.message_conversion)
        format_spec = fstring_expr.children[-2]
        if format_spec.type == 'fstring_format_spec':
            self._check_format_spec(format_spec, depth + 1)

    def is_issue(self, fstring):
        self._check_fstring_contents(fstring.children[1:-1])

    def _check_fstring_contents(self, children, depth=0):
        for fstring_content in children:
            if fstring_content.type == 'fstring_expr':
                self._check_fstring_expr(fstring_content, depth)