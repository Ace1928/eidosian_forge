import itertools
import random
from typing import Any, Dict, FrozenSet, Hashable, Iterable, Mapping, Optional, Set, Tuple, Union
def add_vertices(self, vertices: Iterable[Hashable]) -> None:
    for vertex in vertices:
        self.add_vertex(vertex)