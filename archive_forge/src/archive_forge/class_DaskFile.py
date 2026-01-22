import dask
from distributed.client import Client, _get_global_client
from distributed.worker import Worker
from fsspec import filesystem
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
from fsspec.utils import infer_storage_options
class DaskFile(AbstractBufferedFile):

    def __init__(self, mode='rb', **kwargs):
        if mode != 'rb':
            raise ValueError('Remote dask files can only be opened in "rb" mode')
        super().__init__(**kwargs)

    def _upload_chunk(self, final=False):
        pass

    def _initiate_upload(self):
        """Create remote file/upload"""
        pass

    def _fetch_range(self, start, end):
        """Get the specified set of bytes from remote"""
        return self.fs.fetch_range(self.path, self.mode, start, end)