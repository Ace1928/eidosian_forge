import sys
import unittest
from libcloud.test import LibcloudTestCase
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_S3_PARAMS
from libcloud.storage.drivers.digitalocean_spaces import (
def test_object_enable_cdn_not_implemented(self):
    with self.assertRaises(NotImplementedError):
        self.object.enable_cdn()