import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
def _process_continue_statement(self, node, *loops_to_nodes_of_type):
    try_node, guards = self._get_enclosing_finally_scopes(tuple(loops_to_nodes_of_type))
    if try_node is None:
        raise ValueError('%s that is not enclosed by any of %s' % (node, loops_to_nodes_of_type))
    self.builder.add_continue_node(node, try_node, guards)