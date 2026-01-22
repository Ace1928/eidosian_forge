import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
def _get_enclosing_finally_scopes(self, stop_at):
    included = []
    for node in reversed(self.lexical_scopes):
        if isinstance(node, gast.Try) and node.finalbody:
            included.append(node)
        if isinstance(node, stop_at):
            return (node, included)
    return (None, included)