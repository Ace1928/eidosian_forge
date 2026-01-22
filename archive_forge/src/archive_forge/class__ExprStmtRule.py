import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
@ErrorFinder.register_rule(type='expr_stmt')
class _ExprStmtRule(_CheckAssignmentRule):
    message = 'illegal expression for augmented assignment'
    extended_message = "'{target}' is an " + message

    def is_issue(self, node):
        augassign = node.children[1]
        is_aug_assign = augassign != '=' and augassign.type != 'annassign'
        if self._normalizer.version <= (3, 8) or not is_aug_assign:
            for before_equal in node.children[:-2:2]:
                self._check_assignment(before_equal, is_aug_assign=is_aug_assign)
        if is_aug_assign:
            target = _remove_parens(node.children[0])
            if target.type == 'name' or (target.type in ('atom_expr', 'power') and target.children[1].type == 'trailer' and (target.children[-1].children[0] != '(')):
                return False
            if self._normalizer.version <= (3, 8):
                return True
            else:
                self.add_issue(node, message=self.extended_message.format(target=_get_rhs_name(node.children[0], self._normalizer.version)))