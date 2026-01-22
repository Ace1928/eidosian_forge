from typing import (TYPE_CHECKING, Dict, List, Optional, TextIO, Tuple, Union,
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
import contextlib
import itertools
from . import config as _mod_config
from . import debug, errors, registry, repository
from . import revision as _mod_revision
from . import urlutils
from .controldir import (ControlComponent, ControlComponentFormat,
from .hooks import Hooks
from .inter import InterObject
from .lock import LogicalLockResult
from .revision import RevisionID
from .trace import is_quiet, mutter, mutter_callsite, note, warning
from .transport import Transport, get_transport
def _lefthand_history(self, revision_id, last_rev=None, other_branch=None):
    if 'evil' in debug.debug_flags:
        mutter_callsite(4, '_lefthand_history scales with history.')
    graph = self.repository.get_graph()
    if last_rev is not None:
        if not graph.is_ancestor(last_rev, revision_id):
            raise errors.DivergedBranches(self, other_branch)
    parents_map = graph.get_parent_map([revision_id])
    if revision_id not in parents_map:
        raise errors.NoSuchRevision(self, revision_id)
    current_rev_id = revision_id
    new_history = []
    check_not_reserved_id = _mod_revision.check_not_reserved_id
    while current_rev_id in parents_map and len(parents_map[current_rev_id]) > 0:
        check_not_reserved_id(current_rev_id)
        new_history.append(current_rev_id)
        current_rev_id = parents_map[current_rev_id][0]
        parents_map = graph.get_parent_map([current_rev_id])
    new_history.reverse()
    return new_history