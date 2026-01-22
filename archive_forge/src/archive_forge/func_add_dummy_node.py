from functools import total_ordering
from django.db.migrations.state import ProjectState
from .exceptions import CircularDependencyError, NodeNotFoundError
def add_dummy_node(self, key, origin, error_message):
    node = DummyNode(key, origin, error_message)
    self.node_map[key] = node
    self.nodes[key] = None