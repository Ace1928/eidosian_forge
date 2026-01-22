import collections
import gast
from tensorflow.python.autograph.pyct import gast_util
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
def _ensure_node_in_anf(self, parent, field, node):
    """Puts `node` in A-normal form, by replacing it with a variable if needed.

    The exact definition of A-normal form is given by the configuration.  The
    parent and the incoming field name are only needed because the configuration
    may be context-dependent.

    Args:
      parent: An AST node, the parent of `node`.
      field: The field name under which `node` is the child of `parent`.
      node: An AST node, potentially to be replaced with a variable reference.

    Returns:
      node: An AST node; the argument if transformation was not necessary,
        or the new variable reference if it was.
    """
    if node is None:
        return node
    if _is_trivial(node):
        return node
    if isinstance(node, list):
        return [self._ensure_node_in_anf(parent, field, n) for n in node]
    if isinstance(node, gast.keyword):
        node.value = self._ensure_node_in_anf(parent, field, node.value)
        return node
    if isinstance(node, (gast.Starred, gast.withitem, gast.slice)):
        return self._ensure_fields_in_anf(node, parent, field)
    if self._should_transform(parent, field, node):
        return self._do_transform_node(node)
    else:
        return node