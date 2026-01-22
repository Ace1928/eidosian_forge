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
def _resolve_branch(self, branch_label: str) -> Optional[Revision]:
    try:
        branch_rev = self._revision_map[branch_label]
    except KeyError:
        try:
            nonbranch_rev = self._revision_for_ident(branch_label)
        except ResolutionError as re:
            raise ResolutionError("No such branch: '%s'" % branch_label, branch_label) from re
        else:
            return nonbranch_rev
    else:
        return branch_rev