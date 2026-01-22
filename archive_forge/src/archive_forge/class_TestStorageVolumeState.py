import sys
import unittest
from unittest import TestCase
from libcloud.compute.types import (
class TestStorageVolumeState(TestCase):

    def test_storagevolumestate_tostring(self):
        self.assertEqual(StorageVolumeState.tostring(StorageVolumeState.AVAILABLE), 'AVAILABLE')

    def test_storagevolumestate_fromstring(self):
        self.assertEqual(StorageVolumeState.fromstring('available'), StorageVolumeState.AVAILABLE)