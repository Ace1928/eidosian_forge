import base64
import io
import logging
import smart_open.bytebuffer
import smart_open.constants
class _RawReader(object):
    """Read an Azure Blob Storage file."""

    def __init__(self, blob, size, concurrency):
        self._blob = blob
        self._size = size
        self._position = 0
        self._concurrency = concurrency

    def seek(self, position):
        """Seek to the specified position (byte offset) in the Azure Blob Storage blob.

        :param int position: The byte offset from the beginning of the blob.

        Returns the position after seeking.
        """
        self._position = position
        return self._position

    def read(self, size=-1):
        if self._position >= self._size:
            return b''
        binary = self._download_blob_chunk(size)
        self._position += len(binary)
        return binary

    def _download_blob_chunk(self, size):
        if self._size == self._position:
            return b''
        elif size == -1:
            stream = self._blob.download_blob(offset=self._position, max_concurrency=self._concurrency)
        else:
            stream = self._blob.download_blob(offset=self._position, max_concurrency=self._concurrency, length=size)
        logging.debug('reading with a max concurrency of %d', self._concurrency)
        if isinstance(stream, azure.storage.blob.StorageStreamDownloader):
            binary = stream.readall()
        else:
            binary = stream.read()
        return binary