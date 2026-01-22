import collections
import gast
from tensorflow.python.autograph.pyct import gast_util
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
def _visit_strict_expression(self, node):
    node = self.generic_visit(node)
    self._ensure_fields_in_anf(node)
    return node