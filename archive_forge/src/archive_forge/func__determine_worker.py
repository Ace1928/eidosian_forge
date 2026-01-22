import dask
from distributed.client import Client, _get_global_client
from distributed.worker import Worker
from fsspec import filesystem
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
from fsspec.utils import infer_storage_options
def _determine_worker(self):
    if _in_worker():
        self.worker = True
        if self.fs is None:
            self.fs = filesystem(self.target_protocol, **self.target_options or {})
    else:
        self.worker = False
        self.client = _get_client(self.client)
        self.rfs = dask.delayed(self)