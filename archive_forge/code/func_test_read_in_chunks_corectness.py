import sys
import random
import socket
import string
import os.path
import platform
import unittest
import warnings
from itertools import chain
import pytest
import libcloud.utils.files
from libcloud.utils.py3 import StringIO, b, bchr, urlquote, hexadigits
from libcloud.utils.misc import get_driver, set_driver, get_secure_random_string
from libcloud.common.types import LibcloudError
from libcloud.compute.types import Provider
from libcloud.utils.publickey import get_pubkey_ssh2_fingerprint, get_pubkey_openssh_fingerprint
from libcloud.utils.decorators import wrap_non_libcloud_exceptions
from libcloud.utils.networking import (
from libcloud.compute.providers import DRIVERS
from libcloud.compute.drivers.dummy import DummyNodeDriver
from libcloud.storage.drivers.dummy import DummyIterator
from io import FileIO as file
def test_read_in_chunks_corectness(self):
    data = ''.join((random.choice(string.ascii_lowercase) for i in range(5 * 1024 * 1024))).encode('utf-8')
    chunk_size = None
    iterator = libcloud.utils.files.read_in_chunks(iter([data]), chunk_size=chunk_size, fill_size=False)
    self.assertEqual(data, libcloud.utils.files.exhaust_iterator(iterator))
    chunk_size = 1024 * 1024
    iterator = libcloud.utils.files.read_in_chunks(iter([data]), chunk_size=chunk_size, fill_size=False)
    self.assertEqual(data, libcloud.utils.files.exhaust_iterator(iterator))
    chunk_size = 2 * 1024 * 1024
    iterator = libcloud.utils.files.read_in_chunks(iter([data]), chunk_size=chunk_size, fill_size=False)
    self.assertEqual(data, libcloud.utils.files.exhaust_iterator(iterator))
    chunk_size = 5 * 1024 * 1024
    iterator = libcloud.utils.files.read_in_chunks(iter([data]), chunk_size=chunk_size, fill_size=False)
    self.assertEqual(data, libcloud.utils.files.exhaust_iterator(iterator))
    chunk_size = 10 * 1024 * 1024
    iterator = libcloud.utils.files.read_in_chunks(iter([data]), chunk_size=chunk_size, fill_size=False)
    self.assertEqual(data, libcloud.utils.files.exhaust_iterator(iterator))
    chunk_size = None
    iterator = libcloud.utils.files.read_in_chunks(iter([data]), chunk_size=chunk_size, fill_size=True)
    self.assertEqual(data, libcloud.utils.files.exhaust_iterator(iterator))
    chunk_size = 1024 * 1024
    iterator = libcloud.utils.files.read_in_chunks(iter([data]), chunk_size=chunk_size, fill_size=True)
    self.assertEqual(data, libcloud.utils.files.exhaust_iterator(iterator))
    chunk_size = 2 * 1024 * 1024
    iterator = libcloud.utils.files.read_in_chunks(iter([data]), chunk_size=chunk_size, fill_size=True)
    self.assertEqual(data, libcloud.utils.files.exhaust_iterator(iterator))
    chunk_size = 5 * 1024 * 1024
    iterator = libcloud.utils.files.read_in_chunks(iter([data]), chunk_size=chunk_size, fill_size=True)
    self.assertEqual(data, libcloud.utils.files.exhaust_iterator(iterator))
    chunk_size = 10 * 1024 * 1024
    iterator = libcloud.utils.files.read_in_chunks(iter([data]), chunk_size=chunk_size, fill_size=True)
    self.assertEqual(data, libcloud.utils.files.exhaust_iterator(iterator))