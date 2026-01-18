import collections
import gast
from tensorflow.python.autograph.pyct import gast_util
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
def visit_AsyncWith(self, node):
    msg = 'Nontrivial AsyncWith nodes not supported yet (need to think through the semantics).'
    return self._visit_trivial_only_statement(node, msg)