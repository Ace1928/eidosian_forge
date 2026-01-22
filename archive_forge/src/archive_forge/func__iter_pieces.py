import contextlib
import copy
import itertools
import posixpath as pp
import fasteners
from taskflow import exceptions as exc
from taskflow.persistence import path_based
from taskflow.types import tree
def _iter_pieces(self, path, include_root=False):
    if path == self._root.item:
        parts = []
    else:
        parts = path.split(pp.sep)[1:]
    if include_root:
        parts.insert(0, self._root.item)
    for piece in parts:
        yield piece