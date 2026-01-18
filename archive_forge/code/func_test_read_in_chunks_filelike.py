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
def test_read_in_chunks_filelike(self):

    class FakeFile(file):

        def __init__(self):
            self.remaining = 500

        def read(self, size):
            self.remaining -= 1
            if self.remaining == 0:
                return ''
            return 'b' * (size + 1)
    for index, result in enumerate(libcloud.utils.files.read_in_chunks(FakeFile(), chunk_size=10, fill_size=False)):
        self.assertEqual(result, b('b' * 11))
    self.assertEqual(index, 498)
    for index, result in enumerate(libcloud.utils.files.read_in_chunks(FakeFile(), chunk_size=10, fill_size=True)):
        if index != 548:
            self.assertEqual(result, b('b' * 10))
        else:
            self.assertEqual(result, b('b' * 9))
    self.assertEqual(index, 548)