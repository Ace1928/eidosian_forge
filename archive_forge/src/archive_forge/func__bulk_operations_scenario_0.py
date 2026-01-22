import os
from hashlib import md5
import pytest
from fsspec.implementations.local import LocalFileSystem
from fsspec.tests.abstract.copy import AbstractCopyTests  # noqa
from fsspec.tests.abstract.get import AbstractGetTests  # noqa
from fsspec.tests.abstract.put import AbstractPutTests  # noqa
def _bulk_operations_scenario_0(self, some_fs, some_join, some_path):
    """
        Scenario that is used for many cp/get/put tests. Creates the following
        directory and file structure:

        📁 source
        ├── 📄 file1
        ├── 📄 file2
        └── 📁 subdir
            ├── 📄 subfile1
            ├── 📄 subfile2
            └── 📁 nesteddir
                └── 📄 nestedfile
        """
    source = some_join(some_path, 'source')
    subdir = some_join(source, 'subdir')
    nesteddir = some_join(subdir, 'nesteddir')
    some_fs.makedirs(nesteddir)
    some_fs.touch(some_join(source, 'file1'))
    some_fs.touch(some_join(source, 'file2'))
    some_fs.touch(some_join(subdir, 'subfile1'))
    some_fs.touch(some_join(subdir, 'subfile2'))
    some_fs.touch(some_join(nesteddir, 'nestedfile'))
    return source