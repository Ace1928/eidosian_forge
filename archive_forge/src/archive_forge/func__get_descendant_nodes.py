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
def _get_descendant_nodes(self, targets: Collection[Optional[_RevisionOrBase]], map_: Optional[_RevisionMapType]=None, check: bool=False, omit_immediate_dependencies: bool=False, include_dependencies: bool=True) -> Iterator[Any]:
    if omit_immediate_dependencies:

        def fn(rev: Revision) -> Iterable[str]:
            if rev not in targets:
                return rev._all_nextrev
            else:
                return rev.nextrev
    elif include_dependencies:

        def fn(rev: Revision) -> Iterable[str]:
            return rev._all_nextrev
    else:

        def fn(rev: Revision) -> Iterable[str]:
            return rev.nextrev
    return self._iterate_related_revisions(fn, targets, map_=map_, check=check)