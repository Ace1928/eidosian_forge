import contextlib
import errno
import io
import os
import shutil
import cachetools
import fasteners
from oslo_serialization import jsonutils
from oslo_utils import fileutils
from taskflow import exceptions as exc
from taskflow.persistence import path_based
from taskflow.utils import misc
def _write_to(self, filename, contents):
    contents = misc.binary_encode(contents, encoding=self.backend.encoding)
    with io.open(filename, 'wb') as fp:
        fp.write(contents)
    self.backend.file_cache.pop(filename, None)