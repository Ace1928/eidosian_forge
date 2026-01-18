import sys
import json
from unittest.mock import Mock, call
from libcloud.test import unittest
from libcloud.compute.base import NodeSize, NodeImage, NodeLocation, NodeAuthSSHKey
from libcloud.common.upcloud import (
def test_storage_tier_default_value(self):
    storagedevice = _StorageDevice(self.image, self.size)
    d = storagedevice.to_dict()
    self.assertEqual(d['storage_device'][0]['tier'], 'maxiops')