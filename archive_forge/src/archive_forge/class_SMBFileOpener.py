import datetime
import uuid
from stat import S_ISDIR, S_ISLNK
import smbclient
from .. import AbstractFileSystem
from ..utils import infer_storage_options
class SMBFileOpener:
    """writes to remote temporary file, move on commit"""

    def __init__(self, path, temp, mode, port=445, block_size=-1, **kwargs):
        self.path = path
        self.temp = temp
        self.mode = mode
        self.block_size = block_size
        self.kwargs = kwargs
        self.smbfile = None
        self._incontext = False
        self.port = port
        self._open()

    def _open(self):
        if self.smbfile is None or self.smbfile.closed:
            self.smbfile = smbclient.open_file(self.temp, self.mode, port=self.port, buffering=self.block_size, **self.kwargs)

    def commit(self):
        """Move temp file to definitive on success."""
        smbclient.replace(self.temp, self.path, port=self.port)

    def discard(self):
        """Remove the temp file on failure."""
        smbclient.remove(self.temp, port=self.port)

    def __fspath__(self):
        return self.path

    def __iter__(self):
        return self.smbfile.__iter__()

    def __getattr__(self, item):
        return getattr(self.smbfile, item)

    def __enter__(self):
        self._incontext = True
        return self.smbfile.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        self._incontext = False
        self.smbfile.__exit__(exc_type, exc_value, traceback)