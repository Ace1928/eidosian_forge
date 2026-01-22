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
def _parse_downgrade_target(self, current_revisions: _RevisionIdentifierType, target: _RevisionIdentifierType, assert_relative_length: bool) -> Tuple[Optional[str], Optional[_RevisionOrBase]]:
    """
        Parse downgrade command syntax :target to retrieve the target revision
        and branch label (if any) given the :current_revisions stamp of the
        database.

        Returns a tuple (branch_label, target_revision) where branch_label
        is a string from the command specifying the branch to consider (or
        None if no branch given), and target_revision is a Revision object
        which the command refers to. target_revisions is None if the command
        refers to 'base'. The target may be specified in absolute form, or
        relative to :current_revisions.
        """
    if target is None:
        return (None, None)
    assert isinstance(target, str), 'Expected downgrade target in string form'
    match = _relative_destination.match(target)
    if match:
        branch_label, symbol, relative = match.groups()
        rel_int = int(relative)
        if rel_int >= 0:
            if symbol is None:
                raise RevisionError("Relative revision %s didn't produce %d migrations" % (relative, abs(rel_int)))
            rev = self._walk(symbol, rel_int, branch_label, no_overwalk=assert_relative_length)
            if rev is None:
                raise RevisionError('Walked too far')
            return (branch_label, rev)
        else:
            relative_revision = symbol is None
            if relative_revision:
                if branch_label:
                    cr_tuple = util.to_tuple(current_revisions)
                    symbol_list: Sequence[str]
                    symbol_list = self.filter_for_lineage(cr_tuple, branch_label)
                    if not symbol_list:
                        all_current = cast(Set[Revision], self._get_all_current(cr_tuple))
                        sl_all_current = self.filter_for_lineage(all_current, branch_label)
                        symbol_list = [r.revision if r else r for r in sl_all_current]
                    assert len(symbol_list) == 1
                    symbol = symbol_list[0]
                else:
                    current_revisions = util.to_tuple(current_revisions)
                    if not current_revisions:
                        raise RevisionError("Relative revision %s didn't produce %d migrations" % (relative, abs(rel_int)))
                    if len(set(current_revisions)) > 1:
                        util.warn('downgrade -1 from multiple heads is ambiguous; this usage will be disallowed in a future release.')
                    symbol = current_revisions[0]
                    branch_label = symbol
            rev = self._walk(start=self.get_revision(symbol) if branch_label is None else self.get_revision('%s@%s' % (branch_label, symbol)), steps=rel_int, no_overwalk=assert_relative_length)
            if rev is None:
                if relative_revision:
                    raise RevisionError("Relative revision %s didn't produce %d migrations" % (relative, abs(rel_int)))
                else:
                    raise RevisionError('Walked too far')
            return (branch_label, rev)
    branch_label, _, symbol = target.rpartition('@')
    if not branch_label:
        branch_label = None
    return (branch_label, self.get_revision(symbol))