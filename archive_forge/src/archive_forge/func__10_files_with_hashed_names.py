import os
from hashlib import md5
import pytest
from fsspec.implementations.local import LocalFileSystem
from fsspec.tests.abstract.copy import AbstractCopyTests  # noqa
from fsspec.tests.abstract.get import AbstractGetTests  # noqa
from fsspec.tests.abstract.put import AbstractPutTests  # noqa
def _10_files_with_hashed_names(self, some_fs, some_join, some_path):
    """
        Scenario that is used to check cp/get/put files order when source and
        destination are lists. Creates the following directory and file structure:

        📁 source
        └── 📄 {hashed([0-9])}.txt
        """
    source = some_join(some_path, 'source')
    for i in range(10):
        hashed_i = md5(str(i).encode('utf-8')).hexdigest()
        path = some_join(source, f'{hashed_i}.txt')
        some_fs.pipe(path=path, value=f'{i}'.encode('utf-8'))
    return source