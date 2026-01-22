from __future__ import annotations
import collections
import re
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Deque
from typing import Dict
from typing import FrozenSet
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import overload
from typing import Protocol
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy import util as sqlautil
from .. import util
from ..util import not_none
def _collect_downgrade_revisions(self, upper: _RevisionIdentifierType, lower: _RevisionIdentifierType, inclusive: bool, implicit_base: bool, assert_relative_length: bool) -> Tuple[Set[Revision], Tuple[Optional[_RevisionOrBase], ...]]:
    """
        Compute the set of current revisions specified by :upper, and the
        downgrade target specified by :target. Return all dependents of target
        which are currently active.

        :inclusive=True includes the target revision in the set
        """
    branch_label, target_revision = self._parse_downgrade_target(current_revisions=upper, target=lower, assert_relative_length=assert_relative_length)
    if target_revision == 'base':
        target_revision = None
    assert target_revision is None or isinstance(target_revision, Revision)
    roots: List[Revision]
    if target_revision is None:
        roots = [rev for rev in self._revision_map.values() if rev is not None and rev.down_revision is None]
    elif inclusive:
        roots = [target_revision]
    else:
        roots = [is_revision(rev) for rev in self.get_revisions(target_revision.nextrev)]
    if branch_label and len(roots) > 1:
        ancestors = {rev.revision for rev in self._get_ancestor_nodes([self._resolve_branch(branch_label)], include_dependencies=False)}
        roots = [is_revision(rev) for rev in self.get_revisions({rev.revision for rev in roots}.intersection(ancestors))]
        if len(roots) == 0:
            raise RevisionError('Not a valid downgrade target from current heads')
    heads = self.get_revisions(upper)
    downgrade_revisions = set(self._get_descendant_nodes(roots, include_dependencies=True, omit_immediate_dependencies=False))
    active_revisions = set(self._get_ancestor_nodes(heads, include_dependencies=True))
    downgrade_revisions.intersection_update(active_revisions)
    if implicit_base:
        downgrade_revisions.update(active_revisions.difference(self._get_ancestor_nodes(roots)))
    if target_revision is not None and (not downgrade_revisions) and (target_revision not in heads):
        raise RangeNotAncestorError('Nothing to drop', upper)
    return (downgrade_revisions, heads)