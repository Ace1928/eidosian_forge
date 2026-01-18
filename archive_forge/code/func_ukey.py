import os
import pygit2
from fsspec.spec import AbstractFileSystem
from .memory import MemoryFile
def ukey(self, path, ref=None):
    return self.info(path, ref=ref)['hex']