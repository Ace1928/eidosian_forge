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
def _map_branch_labels(self, revisions: Collection[Revision], map_: _RevisionMapType) -> None:
    for revision in revisions:
        if revision.branch_labels:
            assert revision._orig_branch_labels is not None
            for branch_label in revision._orig_branch_labels:
                if branch_label in map_:
                    map_rev = map_[branch_label]
                    assert map_rev is not None
                    raise RevisionError("Branch name '%s' in revision %s already used by revision %s" % (branch_label, revision.revision, map_rev.revision))
                map_[branch_label] = revision