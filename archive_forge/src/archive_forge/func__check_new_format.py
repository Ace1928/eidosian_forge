from __future__ import annotations
import astroid
from pylint.checkers import BaseChecker
from pylint.checkers import utils
def _check_new_format(self, node, func):
    """ Check the new string formatting """
    if isinstance(node.func, astroid.Attribute) and (not isinstance(node.func.expr, astroid.Const)):
        return
    try:
        strnode = next(func.bound.infer())
    except astroid.InferenceError:
        return
    if not isinstance(strnode, astroid.Const):
        return
    if isinstance(strnode.value, bytes):
        self.add_message('ansible-no-format-on-bytestring', node=node)
        return