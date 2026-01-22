import contextlib
import copy
import itertools
import posixpath as pp
import fasteners
from taskflow import exceptions as exc
from taskflow.persistence import path_based
from taskflow.types import tree
def _fetch_node(self, path, normalized=False):
    if not normalized:
        normed_path = self.normpath(path)
    else:
        normed_path = path
    try:
        return self._reverse_mapping[normed_path]
    except KeyError:
        raise exc.NotFound("Path '%s' not found" % path)