import collections
import gast
from tensorflow.python.autograph.pyct import gast_util
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
def _visit_strict_statement(self, node, children_ok_to_transform=True):
    assert not self._pending_statements
    node = self.generic_visit(node)
    if children_ok_to_transform:
        self._ensure_fields_in_anf(node)
    results = self._consume_pending_statements()
    results.append(node)
    return results