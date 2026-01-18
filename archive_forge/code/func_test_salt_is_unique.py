import sys
import string
import unittest
from unittest.mock import Mock, patch
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.nfsn import NFSNConnection
def test_salt_is_unique(self):
    s1 = self.driver._salt()
    s2 = self.driver._salt()
    self.assertNotEqual(s1, s2)