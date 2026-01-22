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
def _shares_lineage(self, target: Optional[_RevisionOrStr], test_against_revs: Sequence[_RevisionOrStr], include_dependencies: bool=False) -> bool:
    if not test_against_revs:
        return True
    if not isinstance(target, Revision):
        resolved_target = not_none(self._revision_for_ident(target))
    else:
        resolved_target = target
    resolved_test_against_revs = [self._revision_for_ident(test_against_rev) if not isinstance(test_against_rev, Revision) else test_against_rev for test_against_rev in util.to_tuple(test_against_revs, default=())]
    return bool(set(self._get_descendant_nodes([resolved_target], include_dependencies=include_dependencies)).union(self._get_ancestor_nodes([resolved_target], include_dependencies=include_dependencies)).intersection(resolved_test_against_revs))