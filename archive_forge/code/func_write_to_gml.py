from __future__ import annotations
from typing import Any, List, NamedTuple, Optional, Tuple
def write_to_gml(self, path: str) -> None:
    import networkx as nx
    nx.write_gml(self._graph, path)