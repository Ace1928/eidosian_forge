import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.vultr import VultrException
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vultr import VultrNodeDriver, VultrNodeDriverV2
def test_pagination(self):
    images = self.driver.list_images()
    VultrMockHttpV2.type = 'PAGINATED'
    paginated_images = self.driver.list_images()
    self.assertEqual(len(images), len(paginated_images))
    for first, second in zip(images, paginated_images):
        self.assertEqual(first.id, second.id)
        self.assertEqual(first.name, second.name)
        self.assertDictEqual(first.extra, second.extra)