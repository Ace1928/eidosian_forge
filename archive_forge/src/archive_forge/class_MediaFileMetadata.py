from __future__ import annotations
import collections
import threading
from typing import Final
from streamlit.logger import get_logger
from streamlit.runtime.media_file_storage import MediaFileKind, MediaFileStorage
class MediaFileMetadata:
    """Metadata that the MediaFileManager needs for each file it manages."""

    def __init__(self, kind: MediaFileKind=MediaFileKind.MEDIA):
        self._kind = kind
        self._is_marked_for_delete = False

    @property
    def kind(self) -> MediaFileKind:
        return self._kind

    @property
    def is_marked_for_delete(self) -> bool:
        return self._is_marked_for_delete

    def mark_for_delete(self) -> None:
        self._is_marked_for_delete = True