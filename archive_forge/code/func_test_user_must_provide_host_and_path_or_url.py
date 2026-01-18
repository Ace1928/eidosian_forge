import os
import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlparse, parse_qsl, assertRaisesRegex
from libcloud.common.types import ProviderError
from libcloud.compute.base import NodeSize, NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.compute.types import (
from libcloud.compute.providers import get_driver
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudstack import CloudStackNodeDriver, CloudStackAffinityGroupType
def test_user_must_provide_host_and_path_or_url(self):
    expected_msg = 'When instantiating CloudStack driver directly you also need to provide url or host and path argument'
    cls = get_driver(Provider.CLOUDSTACK)
    assertRaisesRegex(self, Exception, expected_msg, cls, 'key', 'secret')
    try:
        cls('key', 'secret', True, 'localhost', '/path')
    except Exception:
        self.fail('host and path provided but driver raised an exception')
    try:
        cls('key', 'secret', url='https://api.exoscale.ch/compute')
    except Exception:
        self.fail('url provided but driver raised an exception')