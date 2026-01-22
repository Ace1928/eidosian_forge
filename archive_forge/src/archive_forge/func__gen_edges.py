from __future__ import annotations
from typing import Any
from typing import Collection
from typing import DefaultDict
from typing import Iterable
from typing import Iterator
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TypeVar
from .. import util
from ..exc import CircularDependencyError
def _gen_edges(edges: DefaultDict[_T, Set[_T]]) -> Set[Tuple[_T, _T]]:
    return {(right, left) for left in edges for right in edges[left]}