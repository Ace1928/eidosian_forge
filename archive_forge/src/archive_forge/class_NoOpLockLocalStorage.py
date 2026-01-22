import os
import time
import errno
import shutil
import tempfile
import threading
from hashlib import sha256
from libcloud.utils.py3 import u, relpath
from libcloud.common.base import Connection
from libcloud.utils.files import read_in_chunks, exhaust_iterator
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
class NoOpLockLocalStorage:

    def __init__(self, path, timeout=5):
        self.path = path
        self.lock_acquire_timeout = timeout

    def __enter__(self):
        return True

    def __exit__(self, type, value, traceback):
        return value