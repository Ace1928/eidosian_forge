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
def _filter_merge_sorted_revisions(self, merge_sorted_revisions, start_revision_id, stop_revision_id, stop_rule):
    """Iterate over an inclusive range of sorted revisions."""
    rev_iter = iter(merge_sorted_revisions)
    if start_revision_id is not None:
        for node in rev_iter:
            rev_id = node.key
            if rev_id != start_revision_id:
                continue
            else:
                rev_iter = itertools.chain(iter([node]), rev_iter)
                break
    if stop_revision_id is None:
        for node in rev_iter:
            rev_id = node.key
            yield (rev_id, node.merge_depth, node.revno, node.end_of_merge)
    elif stop_rule == 'exclude':
        for node in rev_iter:
            rev_id = node.key
            if rev_id == stop_revision_id:
                return
            yield (rev_id, node.merge_depth, node.revno, node.end_of_merge)
    elif stop_rule == 'include':
        for node in rev_iter:
            rev_id = node.key
            yield (rev_id, node.merge_depth, node.revno, node.end_of_merge)
            if rev_id == stop_revision_id:
                return
    elif stop_rule == 'with-merges-without-common-ancestry':
        graph = self.repository.get_graph()
        ancestors = graph.find_unique_ancestors(start_revision_id, [stop_revision_id])
        for node in rev_iter:
            rev_id = node.key
            if rev_id not in ancestors:
                continue
            yield (rev_id, node.merge_depth, node.revno, node.end_of_merge)
    elif stop_rule == 'with-merges':
        stop_rev = self.repository.get_revision(stop_revision_id)
        if stop_rev.parent_ids:
            left_parent = stop_rev.parent_ids[0]
        else:
            left_parent = _mod_revision.NULL_REVISION
        reached_stop_revision_id = False
        revision_id_whitelist = []
        for node in rev_iter:
            rev_id = node.key
            if rev_id == left_parent:
                return
            if not reached_stop_revision_id or rev_id in revision_id_whitelist:
                yield (rev_id, node.merge_depth, node.revno, node.end_of_merge)
                if reached_stop_revision_id or rev_id == stop_revision_id:
                    rev = self.repository.get_revision(rev_id)
                    if rev.parent_ids:
                        reached_stop_revision_id = True
                        revision_id_whitelist.extend(rev.parent_ids)
    else:
        raise ValueError('invalid stop_rule %r' % stop_rule)