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
def _read_from(self, filename):
    mtime = os.path.getmtime(filename)
    cache_info = self.backend.file_cache.setdefault(filename, {})
    if not cache_info or mtime > cache_info.get('mtime', 0):
        with io.open(filename, 'r', encoding=self.backend.encoding) as fp:
            cache_info['data'] = fp.read()
            cache_info['mtime'] = mtime
    return cache_info['data']