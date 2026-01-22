import contextlib
import copy
import itertools
import posixpath as pp
import fasteners
from taskflow import exceptions as exc
from taskflow.persistence import path_based
from taskflow.types import tree
class FakeInode(tree.Node):
    """A in-memory filesystem inode-like object."""

    def __init__(self, item, path, value=None):
        super(FakeInode, self).__init__(item, path=path, value=value)