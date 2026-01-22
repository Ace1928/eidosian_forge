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
def _add_branches(self, revisions: Collection[Revision], map_: _RevisionMapType) -> None:
    for revision in revisions:
        if revision.branch_labels:
            revision.branch_labels.update(revision.branch_labels)
            for node in self._get_descendant_nodes([revision], map_, include_dependencies=False):
                node.branch_labels.update(revision.branch_labels)
            parent = node
            while parent and (not parent._is_real_branch_point) and (not parent.is_merge_point):
                parent.branch_labels.update(revision.branch_labels)
                if parent.down_revision:
                    parent = map_[parent.down_revision]
                else:
                    break