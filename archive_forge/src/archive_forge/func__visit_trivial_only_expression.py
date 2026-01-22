import collections
import gast
from tensorflow.python.autograph.pyct import gast_util
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
def _visit_trivial_only_expression(self, node, msg):
    k = len(self._pending_statements)
    node = self.generic_visit(node)
    self._ensure_fields_in_anf(node)
    if len(self._pending_statements) != k:
        raise ValueError(msg)
    else:
        return node