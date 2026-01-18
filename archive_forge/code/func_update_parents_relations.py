from __future__ import annotations
from typing import Generator, Iterable, Union
from prompt_toolkit.buffer import Buffer
from .containers import (
from .controls import BufferControl, SearchBufferControl, UIControl
def update_parents_relations(self) -> None:
    """
        Update child->parent relationships mapping.
        """
    parents = {}

    def walk(e: Container) -> None:
        for c in e.get_children():
            parents[c] = e
            walk(c)
    walk(self.container)
    self._child_to_parent = parents