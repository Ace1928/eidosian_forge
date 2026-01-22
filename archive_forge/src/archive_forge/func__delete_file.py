from __future__ import annotations
import collections
import threading
from typing import Final
from streamlit.logger import get_logger
from streamlit.runtime.media_file_storage import MediaFileKind, MediaFileStorage
def _delete_file(self, file_id: str) -> None:
    """Delete the given file from storage, and remove its metadata from
        self._files_by_id.

        Thread safety: callers must hold `self._lock`.
        """
    _LOGGER.debug('Deleting File: %s', file_id)
    self._storage.delete_file(file_id)
    del self._file_metadata[file_id]