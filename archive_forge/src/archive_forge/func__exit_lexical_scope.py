import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
def _exit_lexical_scope(self, node):
    leaving_node = self.lexical_scopes.pop()
    assert node == leaving_node