import contextlib
import tempfile
from typing import Type
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from . import decorators, errors, hooks, osutils, registry
from . import revision as _mod_revision
from . import trace, transform
from . import transport as _mod_transport
from . import tree as _mod_tree
@staticmethod
def _remove_external_references(parent_map):
    """Remove references that go outside of the parent map.

        :param parent_map: Something returned from Graph.get_parent_map(keys)
        :return: (filtered_parent_map, child_map, tails)
            filtered_parent_map is parent_map without external references
            child_map is the {parent_key: [child_keys]} mapping
            tails is a list of nodes that do not have any parents in the map
        """
    filtered_parent_map = {}
    child_map = {}
    tails = []
    for key, parent_keys in parent_map.items():
        culled_parent_keys = [p for p in parent_keys if p in parent_map]
        if not culled_parent_keys:
            tails.append(key)
        for parent_key in culled_parent_keys:
            child_map.setdefault(parent_key, []).append(key)
        child_map.setdefault(key, [])
        filtered_parent_map[key] = culled_parent_keys
    return (filtered_parent_map, child_map, tails)