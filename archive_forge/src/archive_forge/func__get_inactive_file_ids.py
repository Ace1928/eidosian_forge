from __future__ import annotations
import collections
import threading
from typing import Final
from streamlit.logger import get_logger
from streamlit.runtime.media_file_storage import MediaFileKind, MediaFileStorage
def _get_inactive_file_ids(self) -> set[str]:
    """Compute the set of files that are stored in the manager, but are
        not referenced by any active session. These are files that can be
        safely deleted.

        Thread safety: callers must hold `self._lock`.
        """
    file_ids = set(self._file_metadata.keys())
    for session_file_ids_by_coord in self._files_by_session_and_coord.values():
        file_ids.difference_update(session_file_ids_by_coord.values())
    return file_ids