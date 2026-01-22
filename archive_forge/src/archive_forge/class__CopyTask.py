from __future__ import unicode_literals
import typing
import threading
from six.moves.queue import Queue
from .copy import copy_file_internal, copy_modified_time
from .errors import BulkCopyFailed
from .tools import copy_file_data
class _CopyTask(_Task):
    """A callable that copies from one file another."""

    def __init__(self, src_file, dst_file):
        self.src_file = src_file
        self.dst_file = dst_file

    def __call__(self):
        try:
            copy_file_data(self.src_file, self.dst_file, chunk_size=1024 * 1024)
        finally:
            try:
                self.src_file.close()
            finally:
                self.dst_file.close()