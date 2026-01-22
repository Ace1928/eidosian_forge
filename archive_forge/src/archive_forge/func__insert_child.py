import contextlib
import copy
import itertools
import posixpath as pp
import fasteners
from taskflow import exceptions as exc
from taskflow.persistence import path_based
from taskflow.types import tree
def _insert_child(self, parent_node, basename, value=None):
    child_path = self.join(parent_node.metadata['path'], basename)
    if child_path.startswith(pp.sep * 2):
        child_path = child_path[1:]
    child_node = FakeInode(basename, child_path, value=value)
    parent_node.add(child_node)
    self._reverse_mapping[child_path] = child_node
    return child_node