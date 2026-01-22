import os
from hashlib import md5
import pytest
from fsspec.implementations.local import LocalFileSystem
from fsspec.tests.abstract.copy import AbstractCopyTests  # noqa
from fsspec.tests.abstract.get import AbstractGetTests  # noqa
from fsspec.tests.abstract.put import AbstractPutTests  # noqa
def _dir_and_file_with_same_name_prefix(self, some_fs, some_join, some_path):
    """
        Scenario that is used to check cp/get/put on directory and file with
        the same name prefixes. Creates the following directory and file structure:

        📁 source
        ├── 📄 subdir.txt
        └── 📁 subdir
            └── 📄 subfile.txt
        """
    source = some_join(some_path, 'source')
    subdir = some_join(source, 'subdir')
    file = some_join(source, 'subdir.txt')
    subfile = some_join(subdir, 'subfile.txt')
    some_fs.makedirs(subdir)
    some_fs.touch(file)
    some_fs.touch(subfile)
    return source