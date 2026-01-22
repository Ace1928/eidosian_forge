from collections import deque
from typing import Callable, Deque, Iterable, List, Optional, Tuple
from .otBase import BaseTable
class SubTablePath(Tuple[BaseTable.SubTableEntry, ...]):

    def __str__(self) -> str:
        path_parts = []
        for entry in self:
            path_part = entry.name
            if entry.index is not None:
                path_part += f'[{entry.index}]'
            path_parts.append(path_part)
        return '.'.join(path_parts)