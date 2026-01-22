import weakref
from weakref import ReferenceType
@staticmethod
def _insert_link(first_node, new_node, last_node):
    LinkedListNode.link_nodes(first_node, new_node)
    LinkedListNode.link_nodes(new_node, last_node)