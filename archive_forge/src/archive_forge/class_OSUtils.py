import os
import math
import functools
import logging
import socket
import threading
import random
import string
import concurrent.futures
from botocore.compat import six
from botocore.vendored.requests.packages.urllib3.exceptions import \
from botocore.exceptions import IncompleteReadError
import s3transfer.compat
from s3transfer.exceptions import RetriesExceededError, S3UploadFailedError
class OSUtils(object):

    def get_file_size(self, filename):
        return os.path.getsize(filename)

    def open_file_chunk_reader(self, filename, start_byte, size, callback):
        return ReadFileChunk.from_filename(filename, start_byte, size, callback, enable_callback=False)

    def open(self, filename, mode):
        return open(filename, mode)

    def remove_file(self, filename):
        """Remove a file, noop if file does not exist."""
        try:
            os.remove(filename)
        except OSError:
            pass

    def rename_file(self, current_filename, new_filename):
        s3transfer.compat.rename_file(current_filename, new_filename)