import os
import sys
import uuid
import warnings
from ftplib import FTP, Error, error_perm
from typing import Any
from ..spec import AbstractBufferedFile, AbstractFileSystem
from ..utils import infer_storage_options, isfilelike
class FTPFile(AbstractBufferedFile):
    """Interact with a remote FTP file with read/write buffering"""

    def __init__(self, fs, path, mode='rb', block_size='default', autocommit=True, cache_type='readahead', cache_options=None, **kwargs):
        super().__init__(fs, path, mode=mode, block_size=block_size, autocommit=autocommit, cache_type=cache_type, cache_options=cache_options, **kwargs)
        if not autocommit:
            self.target = self.path
            self.path = '/'.join([kwargs['tempdir'], str(uuid.uuid4())])

    def commit(self):
        self.fs.mv(self.path, self.target)

    def discard(self):
        self.fs.rm(self.path)

    def _fetch_range(self, start, end):
        """Get bytes between given byte limits

        Implemented by raising an exception in the fetch callback when the
        number of bytes received reaches the requested amount.

        Will fail if the server does not respect the REST command on
        retrieve requests.
        """
        out = []
        total = [0]

        def callback(x):
            total[0] += len(x)
            if total[0] > end - start:
                out.append(x[:end - start - total[0]])
                if end < self.size:
                    raise TransferDone
            else:
                out.append(x)
            if total[0] == end - start and end < self.size:
                raise TransferDone
        try:
            self.fs.ftp.retrbinary(f'RETR {self.path}', blocksize=self.blocksize, rest=start, callback=callback)
        except TransferDone:
            try:
                self.fs.ftp.abort()
                self.fs.ftp.getmultiline()
            except Error:
                self.fs._connect()
        return b''.join(out)

    def _upload_chunk(self, final=False):
        self.buffer.seek(0)
        self.fs.ftp.storbinary(f'STOR {self.path}', self.buffer, blocksize=self.blocksize, rest=self.offset)
        return True