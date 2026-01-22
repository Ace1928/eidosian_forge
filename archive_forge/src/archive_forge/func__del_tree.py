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
def _del_tree(self, path, transaction):
    with self._path_lock(path):
        shutil.rmtree(path)