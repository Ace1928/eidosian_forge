import concurrent.futures
import hashlib
import logging
import sys
from unittest import mock
import fixtures
import os_service_types
import testtools
import openstack
from openstack import exceptions
from openstack.tests.unit import base
from openstack import utils
def test_unicode_strings(self):
    root = 'http://www.example.com'
    leaves = (u'ascii', u'extra_chars-™')
    try:
        result = utils.urljoin(root, *leaves)
    except Exception:
        self.fail('urljoin failed on unicode strings')
    self.assertEqual(result, u'http://www.example.com/ascii/extra_chars-™')