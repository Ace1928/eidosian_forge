from typing import List, Type, TYPE_CHECKING, Optional, Iterable
from .lazy_import import lazy_import
import time
from breezy import (
from breezy.i18n import gettext
from . import controldir, debug, errors, graph, registry, revision as _mod_revision, ui
from .decorators import only_raises
from .inter import InterObject
from .lock import LogicalLockResult, _RelockDebugMixin
from .revisiontree import RevisionTree
from .trace import (log_exception_quietly, mutter, mutter_callsite, note,
def _iter_for_revno(repo, partial_history_cache, stop_index=None, stop_revision=None):
    """Extend the partial history to include a given index

    If a stop_index is supplied, stop when that index has been reached.
    If a stop_revision is supplied, stop when that revision is
    encountered.  Otherwise, stop when the beginning of history is
    reached.

    Args:
      stop_index: The index which should be present.  When it is
        present, history extension will stop.
      stop_revision: The revision id which should be present.  When
        it is encountered, history extension will stop.
    """
    start_revision = partial_history_cache[-1]
    graph = repo.get_graph()
    iterator = graph.iter_lefthand_ancestry(start_revision, (_mod_revision.NULL_REVISION,))
    try:
        next(iterator)
        while True:
            if stop_index is not None and len(partial_history_cache) > stop_index:
                break
            if partial_history_cache[-1] == stop_revision:
                break
            revision_id = next(iterator)
            partial_history_cache.append(revision_id)
    except StopIteration:
        return