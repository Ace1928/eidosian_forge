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
def test_pubkey_openssh_fingerprint(self):
    fp = get_pubkey_openssh_fingerprint(self.PUBKEY)
    self.assertEqual(fp, '35:22:13:5b:82:e2:5d:e1:90:8c:73:74:9f:ef:3b:d8')