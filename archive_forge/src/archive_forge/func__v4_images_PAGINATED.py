import sys
import unittest
from datetime import datetime
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import Node, NodeImage, NodeState, StorageVolume
from libcloud.test.compute import TestCaseMixin
from libcloud.common.linode import LinodeDisk, LinodeIPAddress, LinodeExceptionV4
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.linode import LinodeNodeDriver, LinodeNodeDriverV4
def _v4_images_PAGINATED(self, method, url, body, headers):
    if 'page=2' not in url:
        body = self.fixtures.load('list_images_paginated_page_1.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])
    else:
        body = self.fixtures.load('list_images_paginated_page_2.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])