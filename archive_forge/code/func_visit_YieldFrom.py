import collections
import gast
from tensorflow.python.autograph.pyct import gast_util
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
def visit_YieldFrom(self, node):
    msg = 'Nontrivial YieldFrom nodes not supported yet (need to unit-test them in Python 2).'
    return self._visit_trivial_only_expression(node, msg)