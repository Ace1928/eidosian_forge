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
def test_read_in_chunks_iterator(self):

    def iterator():
        for x in range(0, 1000):
            yield 'aa'
    chunk_count = 0
    for result in libcloud.utils.files.read_in_chunks(iterator(), chunk_size=10, fill_size=False):
        chunk_count += 1
        self.assertEqual(result, b('aa'))
    self.assertEqual(chunk_count, 1000)
    chunk_count = 0
    for result in libcloud.utils.files.read_in_chunks(iterator(), chunk_size=10, fill_size=True):
        chunk_count += 1
        self.assertEqual(result, b('a') * 10)
    self.assertEqual(chunk_count, 200)