import logging
import os
from collections import defaultdict
from concurrent.futures import as_completed, CancelledError, TimeoutError
from copy import deepcopy
from errno import EEXIST, ENOENT
from hashlib import md5
from io import StringIO
from os import environ, makedirs, stat, utime
from os.path import (
from posixpath import join as urljoin
from random import shuffle
from time import time
from threading import Thread
from queue import Queue
from queue import Empty as QueueEmpty
from urllib.parse import quote
import json
from swiftclient import Connection
from swiftclient.command_helpers import (
from swiftclient.utils import (
from swiftclient.exceptions import ClientException
from swiftclient.multithreading import MultiThreadingManager
class _SwiftReader:
    """
    Class for downloading objects from swift and raising appropriate
    errors on failures caused by either invalid md5sum or size of the
    data read.
    """

    def __init__(self, path, body, headers, checksum=True):
        self._path = path
        self._body = body
        self._txn_id = headers.get('x-openstack-request-id')
        if self._txn_id is None:
            self._txn_id = headers.get('x-trans-id')
        self._actual_read = 0
        self._content_length = None
        self._actual_md5 = None
        self._expected_md5 = headers.get('etag', '')
        if len(self._expected_md5) > 1 and self._expected_md5[0] == '"' and (self._expected_md5[-1] == '"'):
            self._expected_md5 = self._expected_md5[1:-1]
        bad_md5_headers = set(['content-range', 'x-object-manifest', 'x-static-large-object'])
        if bad_md5_headers.intersection(headers):
            self._expected_md5 = ''
        if self._expected_md5 and checksum:
            self._actual_md5 = md5()
        if 'content-length' in headers:
            try:
                self._content_length = int(headers.get('content-length'))
            except ValueError:
                raise SwiftError('content-length header must be an integer')

    def __iter__(self):
        for chunk in self._body:
            if self._actual_md5:
                self._actual_md5.update(chunk)
            self._actual_read += len(chunk)
            yield chunk
        self._check_contents()

    def _check_contents(self):
        if self._content_length is not None and self._actual_read != self._content_length:
            raise SwiftError('Error downloading {0}: read_length != content_length, {1:d} != {2:d} (txn: {3})'.format(self._path, self._actual_read, self._content_length, self._txn_id or 'unknown'))
        if self._actual_md5 and self._expected_md5:
            etag = self._actual_md5.hexdigest()
            if etag != self._expected_md5:
                raise SwiftError('Error downloading {0}: md5sum != etag, {1} != {2} (txn: {3})'.format(self._path, etag, self._expected_md5, self._txn_id or 'unknown'))

    def bytes_read(self):
        return self._actual_read