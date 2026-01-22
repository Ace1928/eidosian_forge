from collections import deque
from . import errors, revision
def _find_tips(self):
    return [node for node in self._nodes.values() if not node.child_keys]