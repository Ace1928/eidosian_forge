from fontTools import ttLib
from fontTools.ttLib.tables import otBase
from fontTools.ttLib.tables import otTables as ot
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import deque
from typing import (
def _bfs_base_table(root: otBase.BaseTable, root_accessor: str) -> Iterable[SubTablePath]:
    yield from _traverse_ot_data(root, root_accessor, lambda frontier, new: frontier.extend(new))