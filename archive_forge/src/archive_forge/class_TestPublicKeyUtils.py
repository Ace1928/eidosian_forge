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
class TestPublicKeyUtils(unittest.TestCase):
    PUBKEY = 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDOfbWSXOlqvYjZmRO84/lIoV4gvuX+P1lLg50MMg6jZjLZIlYY081XPRmuom0xY0+BO++J2KgLl7gxJ6xMsKK2VQ+TakdfAH20XfMcTohd/zVCeWsbqZQvEhVXBo4hPIktcfNz0u9Ez3EtInO+kb7raLcRhOVi9QmOkOrCWtQU9mS71AWJuqI9H0YAnTiI8Hs5bn2tpMIqmTXT3g2bwywC25x1Nx9Hy0/FP+KUL6AgvDXv47l+TgSDfTBEkvq+IF1ITrnaOG+nRE02oZC6cwHYTifM/IOollkujxIQmi2Z+j66OHSrjnEQugr0FqGJF2ygKfIh/i2u3fVLM60qE2NN user@example'

    def test_pubkey_openssh_fingerprint(self):
        fp = get_pubkey_openssh_fingerprint(self.PUBKEY)
        self.assertEqual(fp, '35:22:13:5b:82:e2:5d:e1:90:8c:73:74:9f:ef:3b:d8')

    def test_pubkey_ssh2_fingerprint(self):
        fp = get_pubkey_ssh2_fingerprint(self.PUBKEY)
        self.assertEqual(fp, '11:ad:5d:4c:5b:99:c9:80:7e:81:03:76:5a:25:9d:8c')