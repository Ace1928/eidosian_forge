from functools import total_ordering
from django.db.migrations.state import ProjectState
from .exceptions import CircularDependencyError, NodeNotFoundError
class DummyNode(Node):
    """
    A node that doesn't correspond to a migration file on disk.
    (A squashed migration that was removed, for example.)

    After the migration graph is processed, all dummy nodes should be removed.
    If there are any left, a nonexistent dependency error is raised.
    """

    def __init__(self, key, origin, error_message):
        super().__init__(key)
        self.origin = origin
        self.error_message = error_message

    def raise_error(self):
        raise NodeNotFoundError(self.error_message, self.key, origin=self.origin)