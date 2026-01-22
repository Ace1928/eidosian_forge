import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
@ErrorFinder.register_rule(types=_STAR_EXPR_PARENTS)
class _StarExprParentRule(SyntaxRule):

    def is_issue(self, node):

        def is_definition(node, ancestor):
            if ancestor is None:
                return False
            type_ = ancestor.type
            if type_ == 'trailer':
                return False
            if type_ == 'expr_stmt':
                return node.start_pos < ancestor.children[-1].start_pos
            return is_definition(node, ancestor.parent)
        if is_definition(node, node.parent):
            args = [c for c in node.children if c != ',']
            starred = [c for c in args if c.type == 'star_expr']
            if len(starred) > 1:
                if self._normalizer.version < (3, 9):
                    message = 'two starred expressions in assignment'
                else:
                    message = 'multiple starred expressions in assignment'
                self.add_issue(starred[1], message=message)
            elif starred:
                count = args.index(starred[0])
                if count >= 256:
                    message = 'too many expressions in star-unpacking assignment'
                    self.add_issue(starred[0], message=message)