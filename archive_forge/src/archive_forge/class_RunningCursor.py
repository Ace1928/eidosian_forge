from __future__ import annotations
from typing import Any
from streamlit import util
from streamlit.runtime.scriptrunner import get_script_run_ctx
class RunningCursor(Cursor):

    def __init__(self, root_container: int, parent_path: tuple[int, ...]=()):
        """A moving pointer to a delta location in the app.

        RunningCursors auto-increment to the next available location when you
        call get_locked_cursor() on them.

        Parameters
        ----------
        root_container: int
            The root container this cursor lives in.
        parent_path: tuple of ints
          The full path of this cursor, consisting of the IDs of all ancestors.
          The 0th item is the topmost ancestor.

        """
        self._root_container = root_container
        self._parent_path = parent_path
        self._index = 0

    @property
    def root_container(self) -> int:
        return self._root_container

    @property
    def parent_path(self) -> tuple[int, ...]:
        return self._parent_path

    @property
    def index(self) -> int:
        return self._index

    @property
    def is_locked(self) -> bool:
        return False

    def get_locked_cursor(self, **props) -> LockedCursor:
        locked_cursor = LockedCursor(root_container=self._root_container, parent_path=self._parent_path, index=self._index, **props)
        self._index += 1
        return locked_cursor