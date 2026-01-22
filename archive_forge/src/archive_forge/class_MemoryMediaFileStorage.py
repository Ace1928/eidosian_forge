from __future__ import annotations
import contextlib
import hashlib
import mimetypes
import os.path
from typing import Final, NamedTuple
from streamlit.logger import get_logger
from streamlit.runtime.media_file_storage import (
from streamlit.runtime.stats import CacheStat, CacheStatsProvider, group_stats
from streamlit.util import HASHLIB_KWARGS
class MemoryMediaFileStorage(MediaFileStorage, CacheStatsProvider):

    def __init__(self, media_endpoint: str):
        """Create a new MemoryMediaFileStorage instance

        Parameters
        ----------
        media_endpoint
            The name of the local endpoint that media is served from.
            This endpoint should start with a forward-slash (e.g. "/media").
        """
        self._files_by_id: dict[str, MemoryFile] = {}
        self._media_endpoint = media_endpoint

    def load_and_get_id(self, path_or_data: str | bytes, mimetype: str, kind: MediaFileKind, filename: str | None=None) -> str:
        """Add a file to the manager and return its ID."""
        file_data: bytes
        if isinstance(path_or_data, str):
            file_data = self._read_file(path_or_data)
        else:
            file_data = path_or_data
        file_id = _calculate_file_id(file_data, mimetype, filename)
        if file_id not in self._files_by_id:
            _LOGGER.debug('Adding media file %s', file_id)
            media_file = MemoryFile(content=file_data, mimetype=mimetype, kind=kind, filename=filename)
            self._files_by_id[file_id] = media_file
        return file_id

    def get_file(self, filename: str) -> MemoryFile:
        """Return the MemoryFile with the given filename. Filenames are of the
        form "file_id.extension". (Note that this is *not* the optional
        user-specified filename for download files.)

        Raises a MediaFileStorageError if no such file exists.
        """
        file_id = os.path.splitext(filename)[0]
        try:
            return self._files_by_id[file_id]
        except KeyError as e:
            raise MediaFileStorageError(f"Bad filename '{filename}'. (No media file with id '{file_id}')") from e

    def get_url(self, file_id: str) -> str:
        """Get a URL for a given media file. Raise a MediaFileStorageError if
        no such file exists.
        """
        media_file = self.get_file(file_id)
        extension = get_extension_for_mimetype(media_file.mimetype)
        return f'{self._media_endpoint}/{file_id}{extension}'

    def delete_file(self, file_id: str) -> None:
        """Delete the file with the given ID."""
        with contextlib.suppress(KeyError):
            del self._files_by_id[file_id]

    def _read_file(self, filename: str) -> bytes:
        """Read a file into memory. Raise MediaFileStorageError if we can't."""
        try:
            with open(filename, 'rb') as f:
                return f.read()
        except Exception as ex:
            raise MediaFileStorageError(f"Error opening '{filename}'") from ex

    def get_stats(self) -> list[CacheStat]:
        files_by_id = self._files_by_id.copy()
        stats: list[CacheStat] = []
        for file_id, file in files_by_id.items():
            stats.append(CacheStat(category_name='st_memory_media_file_storage', cache_name='', byte_length=len(file.content)))
        return group_stats(stats)